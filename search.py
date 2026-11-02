"""
search.py - retrieval: find the passages most relevant to a question. Three modes:

  vector   embedding similarity (pgvector, cosine). Finds paraphrases and related wording,
           but can rank an exact term or number (an ID, "60.9 months") below vaguer passages.
  keyword  Postgres full-text search (stemmed English words, ts_rank). Finds exact terms,
           names and numbers, but misses passages that say the same thing in other words.
  hybrid   both, merged with Reciprocal Rank Fusion: take the top 20 of each, score every
           passage 1/(60 + rank) per list it appears in, and sort by the sum. Uses only the
           ranks, so the two very different score scales never have to be compared.

Run:  python search.py "What does APOE do to amyloid and tau?"
      python search.py "sex differences in genetic risk" --top-k 8 --show-text --retrieval hybrid

If the right passage is not in the top results here, the LLM cannot fix it later.
The default mode comes from RETRIEVAL in the environment / .env (vector if unset).
"""
import argparse
import os
import textwrap
from typing import Dict, List, Optional, Sequence, Tuple

DSN = os.environ.get("DATABASE_URL", "postgresql://rag:rag@localhost:5433/rag")
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
# BGE models work best when *queries* (not documents) carry this instruction.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

MODES = ("vector", "keyword", "hybrid")
CANDIDATES = 20          # passages each method contributes to the fusion
RRF_K = 60               # the usual RRF constant: damps the weight of the very top ranks

COLUMNS = "chunk_id, title, year, section, page_start, page_end, text"

# Full-text query: the question's words (stemmed, stop words removed) joined with OR.
# plainto/websearch_to_tsquery join them with AND, and a whole question almost never has every
# word in one passage (measured: 2 of 24 test questions matched at all). ts_rank then rewards
# passages that contain more of the words, and rarer ones.
TSQUERY = "replace(plainto_tsquery('english', %(q)s)::text, '&', '|')::tsquery"


def vector_search(query: str, model, conn, k: int) -> List[tuple]:
    q = model.encode(QUERY_PREFIX + query, normalize_embeddings=True)
    return conn.execute(
        """SELECT {cols}, 1 - (embedding <=> %(v)s) AS score
           FROM chunks ORDER BY embedding <=> %(v)s LIMIT %(k)s""".format(cols=COLUMNS),
        {"v": q, "k": k},
    ).fetchall()


def keyword_search(query: str, conn, k: int) -> List[tuple]:
    # normalization 1: divide the rank by 1 + log(passage length), so long passages don't win
    # just by containing more words
    return conn.execute(
        """WITH q AS (SELECT {tsq} AS query)
           SELECT {cols}, ts_rank(tsv, q.query, 1) AS score
           FROM chunks, q WHERE tsv @@ q.query
           ORDER BY score DESC, chunk_id LIMIT %(k)s""".format(cols=COLUMNS, tsq=TSQUERY),
        {"q": query, "k": k},
    ).fetchall()


def rrf(rankings: Sequence[Sequence[str]], k: int = RRF_K) -> List[Tuple[str, float]]:
    """Reciprocal Rank Fusion of several ranked id lists -> [(id, score)], best first.
    Ties (e.g. rank 2 in one list only vs rank 2 in the other only) keep the order in which
    the ids were first seen, so the first list - vector search - breaks them."""
    scores: Dict[str, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: -item[1])   # sorted() is stable


def hybrid_search(query: str, model, conn, k: int) -> List[tuple]:
    vec = vector_search(query, model, conn, CANDIDATES)
    kw = keyword_search(query, conn, CANDIDATES)
    rows = {r[0]: r[:-1] for r in vec + kw}                 # chunk_id -> row without its score
    fused = rrf([[r[0] for r in vec], [r[0] for r in kw]])[:k]
    return [rows[cid] + (score,) for cid, score in fused]


def default_mode() -> str:
    """RETRIEVAL from the environment or .env (read when needed, after .env has been loaded)."""
    return os.environ.get("RETRIEVAL", "vector").strip().lower() or "vector"


def search(query: str, model, conn, top_k: int = 5, mode: Optional[str] = None) -> List[tuple]:
    """Rows of (chunk_id, title, year, section, page_start, page_end, text, score), best first.
    The score depends on the mode (cosine similarity, ts_rank or RRF): only the order matters."""
    mode = mode or default_mode()
    if mode == "vector":
        return vector_search(query, model, conn, top_k)
    if mode == "keyword":
        return keyword_search(query, conn, top_k)
    if mode == "hybrid":
        return hybrid_search(query, model, conn, top_k)
    raise ValueError("retrieval mode must be one of {}, not {!r}".format(MODES, mode))


def main() -> None:
    import psycopg
    from pgvector.psycopg import register_vector
    from sentence_transformers import SentenceTransformer

    import answer  # noqa: F401  (loads .env: RETRIEVAL and the other settings)

    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--retrieval", choices=MODES, default=default_mode())
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--show-text", action="store_true", help="print the full chunk text")
    args = ap.parse_args()

    model = SentenceTransformer(args.model)
    with psycopg.connect(DSN, connect_timeout=5) as conn:
        register_vector(conn)
        for rank, (cid, title, year, section, p0, p1, text, score) in enumerate(
                search(args.query, model, conn, args.top_k, args.retrieval), start=1):
            print("\n#{}  score {:.3f}  {}".format(rank, score, cid))
            print("    {} ({})".format(title[:90], year))
            print("    section: {}   pages {}-{}".format(section, p0, p1))
            body = text if args.show_text else text[:300] + "..."
            print(textwrap.indent(textwrap.fill(body, 96), "    "))


if __name__ == "__main__":
    main()
