"""
search.py - Step 3 of the RAG project: look at what retrieval returns BEFORE adding an LLM.

Run:  python search.py "What does APOE do to amyloid and tau?"
      python search.py "sex differences in genetic risk" --top-k 8 --show-text

If the right passage is not in the top results here, the LLM cannot fix it later.
"""
import argparse
import os
import textwrap

import psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

DSN = os.environ.get("DATABASE_URL", "postgresql://rag:rag@localhost:5433/rag")
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
# BGE models work best when *queries* (not documents) carry this instruction.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def search(query: str, model: SentenceTransformer, conn, top_k: int = 5):
    q = model.encode(QUERY_PREFIX + query, normalize_embeddings=True)
    return conn.execute(
        """SELECT chunk_id, title, year, section, page_start, page_end, text,
                  1 - (embedding <=> %s) AS score
           FROM chunks ORDER BY embedding <=> %s LIMIT %s""",
        (q, q, top_k),
    ).fetchall()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--show-text", action="store_true", help="print the full chunk text")
    args = ap.parse_args()

    model = SentenceTransformer(args.model)
    with psycopg.connect(DSN) as conn:
        register_vector(conn)
        for rank, (cid, title, year, section, p0, p1, text, score) in enumerate(
                search(args.query, model, conn, args.top_k), start=1):
            print("\n#{}  score {:.3f}  {}".format(rank, score, cid))
            print("    {} ({})".format(title[:90], year))
            print("    section: {}   pages {}-{}".format(section, p0, p1))
            body = text if args.show_text else text[:300] + "..."
            print(textwrap.indent(textwrap.fill(body, 96), "    "))


if __name__ == "__main__":
    main()
