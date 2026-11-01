"""evaluate.py: answer matching, verdicts and the summary - with fake retrieval and LLM."""
import pytest

import answer
import evaluate


def test_norm_unifies_dashes_spaces_and_thousands():
    assert evaluate.norm("5,108 Participants − P") == "5108 participants - p"


def test_contains_evidence_accepts_any_listed_passage():
    text = "We found significantly elevated levels of soluble fibrillar oligomers in AD."
    assert evaluate.contains_evidence(text, ["not here", "elevated levels of soluble fibrillar"])
    assert evaluate.contains_evidence(text, "ELEVATED LEVELS")          # single string, any case
    assert not evaluate.contains_evidence(text, None)


def test_score_answer_counts_fact_groups():
    groups = [["(?<!pre)fibrillar oligomer"], [r"\boc\b"]]
    assert evaluate.score_answer("Fibrillar oligomers, detected by OC [1].", groups) == 2
    assert evaluate.score_answer("Prefibrillar oligomers, detected by A11 [1].", groups) == 0


QUESTION = {"id": "L1", "type": "lookup", "question": "How many regions?", "expected": "29",
            "doc_id": "paper", "must_include": [[r"\b29\b"]],
            "evidence": ["Bonferroni correction for 29 regions"]}
ROWS = [
    ("paper:0003", "P", "2026", "Methods", 2, 2, "Unrelated methods text.", 0.80),
    ("paper:0026", "P", "2026", "Tests", 12, 13, "a Bonferroni correction for 29 regions of interest", 0.78),
]


@pytest.fixture
def fake_search(monkeypatch):
    import search
    monkeypatch.setattr(search, "search", lambda q, model, conn, k: ROWS[:k])


def fake_ask(answer_text, cited):
    def _ask(question, conn, embed_model, top_k, llm_model):
        return {"answer": answer_text, "cited": cited,
                "not_found": answer.NOT_FOUND.lower().rstrip(".") in answer_text.lower(),
                "seconds": {"llm": 0.1}, "llm_model": "fake"}
    return _ask


def test_retrieval_only_reports_rank_of_evidence(fake_search):
    res = evaluate.run_one(QUESTION, None, None, top_k=2, llm_model=None, retrieval_only=True)
    assert res["hit_rank"] == 2 and "verdict" not in res


@pytest.mark.parametrize("answer_text, cited, verdict, cited_gold", [
    ("29 regions were used [2].", [2], "correct", True),
    ("29 regions were used [1].", [1], "correct", False),    # right fact, wrong source cited
    ("33 regions were used [2].", [2], "wrong", True),       # right source, misread number
    (answer.NOT_FOUND, [], "refused", False),
])
def test_verdicts(monkeypatch, fake_search, answer_text, cited, verdict, cited_gold):
    monkeypatch.setattr(answer, "ask", fake_ask(answer_text, cited))
    res = evaluate.run_one(QUESTION, None, None, top_k=2, llm_model="fake", retrieval_only=False)
    assert res["verdict"] == verdict and res["cited_gold"] is cited_gold


def test_unanswerable_must_be_refused(monkeypatch, fake_search):
    q = {"id": "U1", "type": "unanswerable", "question": "Dose?", "expected": "-",
         "doc_id": None, "must_include": [], "evidence": None}
    monkeypatch.setattr(answer, "ask", fake_ask(answer.NOT_FOUND, []))
    assert evaluate.run_one(q, None, None, 2, "fake", False)["verdict"] == "correct"
    monkeypatch.setattr(answer, "ask", fake_ask("10 mg/kg [1].", [1]))
    assert evaluate.run_one(q, None, None, 2, "fake", False)["verdict"].startswith("wrong")


def test_retry_on_rate_limit_then_success(monkeypatch):
    calls = {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise answer.LLMError("LLM request failed (429): quota")
        return {"answer": "ok"}

    monkeypatch.setattr(answer, "ask", flaky)
    monkeypatch.setattr(evaluate.time, "sleep", lambda s: None)
    assert evaluate.ask_with_retry("q", None, None, 5, "m") == {"answer": "ok"}
    assert calls["n"] == 2


def test_summary_counts_per_type():
    results = [
        {"type": "lookup", "hit_rank": 1, "verdict": "correct", "cited_gold": True, "seconds_llm": 1.0},
        {"type": "lookup", "hit_rank": 2, "verdict": "wrong", "cited_gold": True, "seconds_llm": 3.0},
        {"type": "unanswerable", "hit_rank": None, "verdict": "correct", "seconds_llm": 2.0},
    ]
    s = evaluate.summarise(results, top_k=5, retrieval_only=False)
    assert s["lookup"]["hit@5"] == 2 and s["lookup"]["hit@1"] == 1
    assert s["lookup"]["MRR"] == 0.75 and s["lookup"]["correct"] == 1 and s["lookup"]["wrong"] == 1
    assert s["unanswerable"]["correct"] == 1
