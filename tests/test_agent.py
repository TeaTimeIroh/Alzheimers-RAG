"""agent.py: the tool loop, passage numbering, citations and the step limit - with a scripted
fake LLM and fake search, so no model, database or server is needed."""
import json

import pytest

import agent
import answer
import search

PAPERS = [{"id": "bhad369", "title": "Beijing amyloid study", "year": "2023", "passages": 30},
          {"id": "ALZ-21-e14499", "title": "Entorhinal tau study", "year": "2025", "passages": 20}]
CHUNKS = {
    "bhad369": [("bhad369:0006", "Beijing amyloid study", "2023", "Methods", 3, 3,
                 "All participants completed florbetapir PET.", 0.03)],
    "ALZ-21-e14499": [("ALZ-21-e14499:0009", "Entorhinal tau study", "2025", "Methods", 4, 4,
                       "For ADNI we utilized Florbetapir (AV45).", 0.03)],
}


def call(name, **args):
    return {"id": "c-" + name, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


@pytest.fixture
def world(monkeypatch):
    """Fake papers + search; `script` is the list of assistant messages the 'LLM' returns."""
    state = {"script": [], "seen": [], "searches": []}

    def fake_search(query, model, conn, k, mode=None, max_per_paper=None, doc_id=None):
        state["searches"].append((query, doc_id))
        if doc_id:
            return CHUNKS[doc_id][:k]
        return (CHUNKS["ALZ-21-e14499"] + CHUNKS["bhad369"])[:1]   # global search: one paper only

    def fake_chat(messages, model, tools=None):
        state["seen"].append({"messages": [dict(m) for m in messages], "tools": tools})
        return state["script"].pop(0)

    monkeypatch.setattr(agent, "list_papers", lambda conn: PAPERS)
    monkeypatch.setattr(search, "search", fake_search)
    monkeypatch.setattr(agent, "chat_with_retry", fake_chat)
    return state


def test_agent_searches_each_paper_and_cites_both(world):
    world["script"] = [
        {"content": None, "tool_calls": [
            call("search_paper", paper_id="bhad369", query="amyloid PET tracer"),
            call("search_paper", paper_id="ALZ-21-e14499", query="ADNI tracer")]},
        {"content": "Both used florbetapir [1][2]."},
    ]
    r = agent.ask_agent("Which tracer did both use?", None, None, llm_model="m")
    assert r["answer"] == "Both used florbetapir [1][2]." and r["cited"] == [1, 2]
    assert [s["chunk_id"] for s in r["sources"]] == ["bhad369:0006", "ALZ-21-e14499:0009"]
    assert r["steps"] == 1 and r["llm_calls"] == 2 and r["not_found"] is False
    assert world["searches"] == [("amyloid PET tracer", "bhad369"), ("ADNI tracer", "ALZ-21-e14499")]
    # the tool results went back to the model, numbered, with the call ids it used
    tool_msgs = [m for m in world["seen"][1]["messages"] if m["role"] == "tool"]
    assert tool_msgs[0]["tool_call_id"] == "c-search_paper" and tool_msgs[0]["content"].startswith("[1] ")
    # the paper list is in the system prompt
    assert "bhad369 | Beijing amyloid study | 2023" in world["seen"][0]["messages"][0]["content"]


def test_passages_keep_their_number_when_seen_again(world):
    world["script"] = [
        {"tool_calls": [call("search_paper", paper_id="bhad369", query="tracer")]},
        {"tool_calls": [call("search_paper", paper_id="bhad369", query="PET")]},
        {"content": "Florbetapir [1]."},
    ]
    r = agent.ask_agent("q", None, None, llm_model="m")
    assert len(r["sources"]) == 1 and r["steps"] == 2
    second = [m for m in world["seen"][2]["messages"] if m["role"] == "tool"][-1]["content"]
    assert second.startswith("[1] ") and "already shown" in second


def test_bad_tool_calls_are_reported_back_not_crashing(world):
    world["script"] = [
        {"tool_calls": [call("search_paper", paper_id="nope", query="x"), call("delete_db"),
                        {"id": "c3", "function": {"name": "search", "arguments": "{not json"}}]},
        {"tool_calls": [call("search_paper", paper_id="bhad369", query="tracer")]},
        {"content": "Florbetapir [1]."},
    ]
    r = agent.ask_agent("q", None, None, llm_model="m")
    errors = [t.get("error", "") for t in r["trace"]]
    assert "unknown paper_id" in errors[0] and "unknown tool" in errors[1] and "non-empty query" in errors[2]
    assert r["answer"] == "Florbetapir [1]." and r["searched"] is True


def test_step_limit_forces_an_answer_without_tools(world):
    world["script"] = [{"tool_calls": [call("search", query="q{}".format(i))]} for i in range(2)]
    world["script"].append({"content": "Florbetapir [1]."})
    r = agent.ask_agent("q", None, None, llm_model="m", max_steps=2)
    assert r["llm_calls"] == 3 and r["answer"] == "Florbetapir [1]."
    assert world["seen"][-1]["tools"] is None                      # no tools offered at the limit
    assert "Search limit reached" in world["seen"][-1]["messages"][-1]["content"]


def test_thinking_is_stripped_and_empty_answer_counts_as_refusal(world):
    world["script"] = [{"tool_calls": [call("search", query="q")]}, {"content": "<think>hmm</think>"}]
    r = agent.ask_agent("q", None, None, llm_model="m")
    assert r["answer"] == answer.NOT_FOUND and r["not_found"] is True


def test_check_tools(monkeypatch, capsys):
    monkeypatch.setattr(agent, "chat_message", lambda *a, **k: {"tool_calls": [
        {"function": {"name": "add", "arguments": '{"a": 17, "b": 25}'}}]})
    assert agent.check_tools("m") is True
    monkeypatch.setattr(agent, "chat_message", lambda *a, **k: {"content": "42"})
    assert agent.check_tools("m") is False
    assert "NO TOOL CALL" in capsys.readouterr().out


def test_retry_on_rate_limit(monkeypatch):
    calls = {"n": 0}

    def flaky(messages, model, tools=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise answer.LLMError("LLM request failed (429): quota")
        return {"content": "ok"}

    monkeypatch.setattr(agent, "chat_message", flaky)
    monkeypatch.setattr(agent.time, "sleep", lambda s: None)
    assert agent.chat_with_retry([], "m", None) == {"content": "ok"} and calls["n"] == 2


def test_cannot_answer_is_a_structured_refusal(world):
    world["script"] = [
        {"tool_calls": [call("search", query="lecanemab dose")]},
        {"tool_calls": [call("cannot_answer", reason="No paper mentions lecanemab.")]},
    ]
    r = agent.ask_agent("Dose of lecanemab?", None, None, llm_model="m")
    assert r["not_found"] is True and r["gave_up"] is True
    assert r["answer"] == answer.NOT_FOUND + " No paper mentions lecanemab."
    assert r["steps"] == 1 and r["llm_calls"] == 2
    assert "cannot_answer(reason)" in world["seen"][0]["messages"][0]["content"]


def test_answering_without_searching_is_sent_back(world):
    world["script"] = [
        {"content": answer.NOT_FOUND},                                    # refuses blind
        {"tool_calls": [call("search_paper", paper_id="bhad369", query="tracer")]},
        {"content": "Florbetapir [1]."},
    ]
    r = agent.ask_agent("q", None, None, llm_model="m")
    assert r["answer"] == "Florbetapir [1]." and r["nudges"] == 1 and r["not_found"] is False
    assert world["seen"][1]["messages"][-1]["content"] == agent.SEARCH_FIRST


def test_cannot_answer_before_any_search_is_rejected(world):
    world["script"] = [
        {"tool_calls": [call("cannot_answer", reason="probably not there")]},
        {"tool_calls": [call("search", query="q"), call("cannot_answer", reason="not there")]},
    ]
    r = agent.ask_agent("q", None, None, llm_model="m")
    assert "search" in r["trace"][0]["error"].lower()
    assert r["gave_up"] is True and r["searched"] is True                # same-turn search counts


def test_never_searching_ends_at_the_step_limit(world):
    world["script"] = [{"content": "I think it is X."}] * 3
    r = agent.ask_agent("q", None, None, llm_model="m", max_steps=2)
    assert r["searched"] is False and r["nudges"] == 2 and r["answer"] == "I think it is X."
