"""search.py against a real Postgres: the full-text column, the OR query and the hybrid SQL.

Skipped unless TEST_DATABASE_URL is set. CI starts a pgvector/pgvector:pg16 service for it.
Locally (with docker compose up):  set TEST_DATABASE_URL=postgresql://rag:rag@localhost:5433/rag
Everything happens in a separate `rag_test` schema that is dropped afterwards, so the real
`chunks` table is never touched.
"""
import os

import numpy as np
import psycopg
import pytest

import embed_store
import search

DSN = os.environ.get("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not DSN, reason="TEST_DATABASE_URL not set")

DIM = 4
CHUNKS = [  # (chunk_id, section, text, embedding)
    ("a:0001", "Methods", "The mean interval between diagnosis and autopsy was 60.9 months.",
     [1, 0, 0, 0]),
    ("a:0002", "Results", "Atrophy was mainly temporoparietal; frontal regions were relatively spared.",
     [0, 1, 0, 0]),
    ("b:0001", "Methods", "All participants completed florbetapir PET scanning.", [0, 0, 1, 0]),
    ("b:0002", "Discussion", "Amyloid deposition predicts later cognitive decline.", [0, 0, 0, 1]),
]


class FakeModel:
    """Stands in for the embedding model: every query lands on chunk b:0002's vector."""
    def encode(self, text, normalize_embeddings=True):
        return np.array([0, 0, 0.2, 1], dtype=np.float32)


@pytest.fixture(scope="module")
def conn():
    with psycopg.connect(DSN, autocommit=True) as c:
        has_vector = c.execute(
            "SELECT 1 FROM pg_available_extensions WHERE name = 'vector'").fetchone()
        c.execute("DROP SCHEMA IF EXISTS rag_test CASCADE")
        c.execute("CREATE SCHEMA rag_test")
        c.execute("SET search_path TO rag_test, public")
        if has_vector:
            embed_store.create_schema(c, DIM)          # the real schema, as embed_store creates it
            rows = [(cid, cid.split(":")[0], "Paper " + cid[0], sec, txt, np.array(v, dtype=np.float32))
                    for cid, sec, txt, v in CHUNKS]
            c.cursor().executemany(
                "INSERT INTO chunks (chunk_id, doc_id, title, section, text, embedding) "
                "VALUES (%s,%s,%s,%s,%s,%s)", rows)
        else:                                          # plain Postgres: keyword search only
            c.execute("CREATE TABLE chunks (chunk_id text PRIMARY KEY, doc_id text, title text, "
                      "year text, section text, page_start int, page_end int, text text NOT NULL)")
            embed_store.add_fulltext(c)
            c.cursor().executemany(
                "INSERT INTO chunks (chunk_id, doc_id, title, section, text) VALUES (%s,%s,%s,%s,%s)",
                [(cid, cid.split(":")[0], "Paper " + cid[0], sec, txt) for cid, sec, txt, _ in CHUNKS])
        c.has_vector = bool(has_vector)
        yield c
        c.execute("DROP SCHEMA rag_test CASCADE")


def test_fulltext_column_is_filled_by_postgres(conn):
    assert conn.execute("SELECT count(*) FROM chunks WHERE tsv IS NOT NULL").fetchone()[0] == 4


def test_keyword_search_ranks_the_exact_wording_first(conn):
    rows = search.keyword_search("How long was the interval between diagnosis and autopsy?", conn, 5)
    assert rows[0][0] == "a:0001"


def test_keyword_query_matches_any_word_not_all(conn):
    # "sparing" and "spared" share a stem; "brain" and "lobe" appear nowhere - an AND query
    # would return nothing, the OR query still finds the passage
    rows = search.keyword_search("Was the frontal lobe spared in the brain?", conn, 5)
    assert [r[0] for r in rows] == ["a:0002"]


@pytest.mark.parametrize("question", ["what is the", "", "O'Brien & \"tau\" | (!x) <-> 50%"])
def test_keyword_search_survives_stop_words_and_operators(conn, question):
    search.keyword_search(question, conn, 5)           # no syntax error, possibly no rows


def test_hybrid_merges_vector_and_keyword_results(conn):
    if not conn.has_vector:
        pytest.skip("pgvector not installed in this Postgres")
    from pgvector.psycopg import register_vector
    register_vector(conn)
    question = "Which tracer did the participants complete PET with?"
    vec = [r[0] for r in search.vector_search(question, FakeModel(), conn, 4)]
    kw = [r[0] for r in search.keyword_search(question, conn, 4)]
    assert vec[0] == "b:0002" and kw[0] == "b:0001"     # each method has a different favourite
    rows = search.search(question, FakeModel(), conn, 2, "hybrid")
    assert {r[0] for r in rows} == {"b:0001", "b:0002"}  # hybrid keeps both
    assert all(r[7] > 0 for r in rows)
