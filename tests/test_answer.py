"""answer.py: citation parsing, prompt building, the LLM client and ask() - with a fake LLM."""
import pytest
import requests

import answer


class FakeResponse:
    def __init__(self, payload, status=200):
        self._payload, self.status_code, self.text = payload, status, str(payload)

    def json(self):
        return self._payload


def chat_reply(text):
    return FakeResponse({"choices": [{"message": {"content": text}}]})


ROWS = [  # what search.search() returns: id, title, year, section, p0, p1, text, score
    ("paper:0001", "Paper A", "2020", "Results", 3, 4, "29 regions of interest were used.", 0.81),
    ("paper:0002", "Paper A", "2020", "Methods", 2, 2, "Bonferroni correction was applied.", 0.79),
]


@pytest.fixture
def fake_search(monkeypatch):
    import search
    monkeypatch.setattr(search, "search", lambda q, model, conn, k, mode=None, cap=None: ROWS[:k])


# ----------------------------------------------------------------------------- citations
@pytest.mark.parametrize("text, n, expected", [
    ("Claim [2].", 5, [2]),
    ("A [1][3] and B [2, 4].", 5, [1, 2, 3, 4]),
    ("Range [1-3] and en dash [4–5].", 5, [1, 2, 3, 4, 5]),
    ("Out of range [9] and a year [2024].", 5, []),
    ("No citations at all.", 5, []),
])
def test_cited_numbers(text, n, expected):
    assert answer.cited_numbers(text, n) == expected


def test_build_messages_numbers_sources_with_metadata():
    hits = [{"n": i, "title": r[1], "year": r[2], "section": r[3], "page_start": r[4],
             "page_end": r[5], "text": r[6]} for i, r in enumerate(ROWS, start=1)]
    system, user = answer.build_messages("How many regions?", hits)
    assert system["role"] == "system" and answer.NOT_FOUND in system["content"]
    assert "[1] Paper A (2020) - section: Results, pages 3-4" in user["content"]
    assert "[2] Paper A (2020) - section: Methods, pages 2-2" in user["content"]
    assert user["content"].endswith("Question: How many regions?")


# ----------------------------------------------------------------------------- LLM client
def test_chat_strips_reasoning_and_sends_temperature_zero(monkeypatch):
    sent = {}

    def fake_post(url, headers, json, timeout):
        sent.update(url=url, auth=headers["Authorization"], body=json)
        return chat_reply("<think>private reasoning</think> The answer [1].")

    monkeypatch.setattr(answer, "LLM_BASE_URL", "http://llm.test/v1")
    monkeypatch.setattr(answer, "LLM_API_KEY", "secret")
    monkeypatch.setattr(answer.requests, "post", fake_post)
    assert answer.chat([{"role": "user", "content": "q"}], "model-x") == "The answer [1]."
    assert sent["url"] == "http://llm.test/v1/chat/completions"
    assert sent["auth"] == "Bearer secret"
    assert sent["body"]["temperature"] == 0.0 and sent["body"]["model"] == "model-x"


def test_chat_reports_unreachable_server(monkeypatch):
    def refuse(*args, **kwargs):
        raise requests.ConnectionError("refused")
    monkeypatch.setattr(answer.requests, "post", refuse)
    with pytest.raises(answer.LLMError, match="Cannot reach"):
        answer.chat([], "model-x")


def test_chat_reports_http_errors(monkeypatch):
    monkeypatch.setattr(answer.requests, "post",
                        lambda *a, **k: FakeResponse({"error": "quota"}, status=429))
    with pytest.raises(answer.LLMError, match=r"\(429\)"):
        answer.chat([], "model-x")


# ----------------------------------------------------------------------------- ask()
def test_ask_returns_answer_sources_and_citations(monkeypatch, fake_search):
    monkeypatch.setattr(answer, "chat", lambda messages, model: "29 regions were used [1].")
    result = answer.ask("How many regions?", conn=None, embed_model=None, top_k=2, llm_model="m")
    assert result["answer"] == "29 regions were used [1]."
    assert result["cited"] == [1] and result["not_found"] is False
    assert [s["chunk_id"] for s in result["sources"]] == ["paper:0001", "paper:0002"]
    assert "text" not in result["sources"][0]          # sources are metadata only


def test_ask_detects_refusal(monkeypatch, fake_search):
    monkeypatch.setattr(answer, "chat", lambda messages, model: answer.NOT_FOUND)
    result = answer.ask("Dose of lecanemab?", conn=None, embed_model=None, top_k=2, llm_model="m")
    assert result["not_found"] is True and result["cited"] == []


def test_dotenv_does_not_override_real_environment(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("RAG_TEST_A=from-file\nRAG_TEST_B='quoted'\n# comment\n", encoding="utf-8")
    monkeypatch.setenv("RAG_TEST_A", "from-environment")
    monkeypatch.delenv("RAG_TEST_B", raising=False)
    answer.load_dotenv(env)
    import os
    assert os.environ["RAG_TEST_A"] == "from-environment"
    assert os.environ["RAG_TEST_B"] == "quoted"
