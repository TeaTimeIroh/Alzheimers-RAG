"""api.py: endpoints, validation and error mapping - with a fake database, search and LLM."""
import sys
import types

import psycopg
import pytest
from fastapi.testclient import TestClient

import answer
import api

ROWS = [("paper:0026", "Paper", "2026", "Tests", 12, 13, "29 regions of interest", 0.81),
        ("paper:0024", "Paper", None, None, 12, 12, "loci text", 0.79)]


class FakeConn:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, *args):
        class Result:
            def fetchone(self):
                return (473, 19)
        return Result()


@pytest.fixture
def client(monkeypatch):
    state = {"db": "up"}

    def fake_connect(dsn, **kwargs):
        if state["db"] == "down":
            raise psycopg.OperationalError("connection refused\ndetails")
        return FakeConn()

    def fake_ask(question, conn, embed_model, top_k, llm_model=None, retrieval=None):
        state["retrieval"] = retrieval
        if question == "llm down please":
            raise answer.LLMError("Cannot reach http://localhost:1234/v1")
        sources = [{"n": i, "chunk_id": r[0], "title": r[1], "year": r[2], "section": r[3],
                    "page_start": r[4], "page_end": r[5], "score": r[7]}
                   for i, r in enumerate(ROWS[:top_k], start=1)]
        return {"question": question, "answer": "29 regions [1].", "not_found": False,
                "cited": [1], "sources": sources, "llm_model": "fake",
                "retrieval": retrieval or "vector",
                "seconds": {"retrieval": 0.01, "llm": 0.5}, "messages": []}

    # never load the real embedding model in tests, even where it is installed
    stub = types.ModuleType("sentence_transformers")
    stub.SentenceTransformer = lambda name: "embedding-model:" + name
    monkeypatch.setitem(sys.modules, "sentence_transformers", stub)
    monkeypatch.setattr(api.psycopg, "connect", fake_connect)
    monkeypatch.setattr(api, "register_vector", lambda conn: None)
    def fake_search(q, model, conn, k, mode=None):
        state["search_mode"] = mode
        return ROWS[:k]

    monkeypatch.setattr(api, "search", fake_search)
    monkeypatch.setattr(answer, "ask", fake_ask)
    with TestClient(api.app) as c:
        c.state = state
        yield c


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["chunks"] == 473 and body["papers"] == 19
    assert "key" not in str(body).lower()            # never leak the API key


def test_ask_marks_cited_sources(client):
    r = client.post("/ask", json={"question": "How many regions?", "top_k": 2})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "29 regions [1]."
    assert [s["cited"] for s in body["sources"]] == [True, False]
    assert body["sources"][1]["year"] == "" and body["sources"][1]["section"] == ""


def test_search_returns_ranked_chunks(client):
    body = client.get("/search", params={"q": "regions of interest", "k": 1}).json()
    assert len(body) == 1 and body[0]["rank"] == 1 and body[0]["chunk_id"] == "paper:0026"


@pytest.mark.parametrize("payload", [{"question": "hi"}, {"question": "valid question", "top_k": 99}])
def test_ask_validates_input(client, payload):
    assert client.post("/ask", json=payload).status_code == 422


def test_llm_failure_is_502(client):
    r = client.post("/ask", json={"question": "llm down please"})
    assert r.status_code == 502 and "LLM backend error" in r.json()["detail"]


def test_database_failure_is_503(client):
    client.state["db"] = "down"
    r = client.get("/health")
    assert r.status_code == 503 and r.json()["detail"] == "database unavailable: connection refused"


def test_openapi_docs_available(client):
    assert client.get("/openapi.json").status_code == 200


def test_search_mode_is_passed_through(client):
    assert client.get("/search", params={"q": "regions", "mode": "hybrid"}).status_code == 200
    assert client.state["search_mode"] == "hybrid"
    client.get("/search", params={"q": "regions"})
    assert client.state["search_mode"] is None          # None = the server's RETRIEVAL setting


def test_ask_retrieval_is_passed_through_and_reported(client):
    r = client.post("/ask", json={"question": "How many regions?", "retrieval": "keyword"})
    assert r.status_code == 200 and r.json()["retrieval"] == "keyword"
    assert client.state["retrieval"] == "keyword"


def test_unknown_mode_is_rejected(client):
    assert client.get("/search", params={"q": "regions", "mode": "bm25"}).status_code == 422
    assert client.post("/ask", json={"question": "How many?", "retrieval": "bm25"}).status_code == 422
