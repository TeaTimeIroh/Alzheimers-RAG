"""
embed_store.py - Step 2 of the RAG project: chunks.jsonl -> embeddings -> Postgres (pgvector).

  1. reads chunks.jsonl (from chunk_docs.py),
  2. embeds each chunk's `embed_text` with a local sentence-transformers model (no API, no cost),
  3. creates the `chunks` table and upserts every chunk with its embedding.

Re-running is safe: chunks are matched by chunk_id and overwritten, and rows whose chunk_id
is no longer in chunks.jsonl are deleted, so the table always mirrors the file.
Use the same --model in search.py, otherwise query and chunk vectors live in different spaces.

Setup:   pip install -r requirements.txt
         docker compose up -d
Run:     python embed_store.py --chunks chunks.jsonl
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

DSN = os.environ.get("DATABASE_URL", "postgresql://rag:rag@localhost:5433/rag")
CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"      # 384 dimensions, ~130 MB, fine on a laptop CPU


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=Path, default=Path("chunks.jsonl"))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    chunks = [json.loads(line) for line in open(args.chunks, encoding="utf-8") if line.strip()]
    # Postgres text columns can't hold NUL bytes; replace any control characters that
    # slipped through from the PDF (parse_pdfs.py already does this for new output).
    for c in chunks:
        for k, v in c.items():
            if isinstance(v, str):
                c[k] = CONTROL_CHARS.sub("�", v)
    if not chunks:
        sys.exit("No chunks found in {}".format(args.chunks))
    print("{} chunks loaded".format(len(chunks)))

    # Check the database first, so a connection problem doesn't cost a 40 s embedding run.
    try:
        psycopg.connect(DSN, connect_timeout=5).close()
    except psycopg.OperationalError as e:
        sys.exit("Cannot connect to Postgres at {}\n{}\n"
                 "Is the container running (docker compose ps)? Is something else using that port "
                 "(netstat -ano | findstr :5433)?".format(DSN, e))

    print("loading model {} (first run downloads it) ...".format(args.model))
    model = SentenceTransformer(args.model)
    dim = model.get_sentence_embedding_dimension()

    t0 = time.time()
    vectors = model.encode(
        [c["embed_text"] for c in chunks],
        batch_size=args.batch_size,
        normalize_embeddings=True,          # unit vectors: cosine distance == dot-product ranking
        show_progress_bar=True,
    )
    print("embedded in {:.0f}s, vector size {}".format(time.time() - t0, dim))

    with psycopg.connect(DSN, autocommit=True) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(conn)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id   text PRIMARY KEY,
                doc_id     text NOT NULL,
                title      text,
                year       text,
                pmcid      text,
                doi        text,
                section    text,
                page_start int,
                page_end   int,
                n_words    int,
                text       text NOT NULL,
                embedding  vector({dim})
            )""".format(dim=dim))
        conn.execute("CREATE INDEX IF NOT EXISTS chunks_doc_idx ON chunks (doc_id)")
        # An index only pays off with many thousands of rows, but this is how you would add one:
        conn.execute("CREATE INDEX IF NOT EXISTS chunks_emb_idx ON chunks "
                     "USING hnsw (embedding vector_cosine_ops)")
        with conn.cursor() as cur:
            for c, v in zip(chunks, vectors):
                cur.execute("""
                    INSERT INTO chunks (chunk_id, doc_id, title, year, pmcid, doi, section,
                                        page_start, page_end, n_words, text, embedding)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (chunk_id) DO UPDATE SET
                        title=EXCLUDED.title, section=EXCLUDED.section, text=EXCLUDED.text,
                        page_start=EXCLUDED.page_start, page_end=EXCLUDED.page_end,
                        n_words=EXCLUDED.n_words, embedding=EXCLUDED.embedding""",
                    (c["chunk_id"], c["doc_id"], c["title"], c.get("year"), c.get("pmcid"),
                     c.get("doi"), c["section"], c["page_start"], c["page_end"],
                     c["n_words"], c["text"], v))
        # Keep the table identical to chunks.jsonl: remove rows of papers (or chunks) that
        # are no longer in the file, e.g. after deleting a PDF and re-running the chunker.
        removed = conn.execute("DELETE FROM chunks WHERE NOT (chunk_id = ANY(%s))",
                               ([c["chunk_id"] for c in chunks],)).rowcount
        n = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
    print("done - table `chunks` now has {} rows ({} stale rows removed)".format(n, removed))


if __name__ == "__main__":
    main()
