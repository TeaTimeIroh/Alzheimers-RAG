"""evaluate.py: answer matching, verdicts and the summary - with fake retrieval and LLM."""
import json

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
    monkeypatch.setattr(search, "search", lambda q, model, conn, k, mode=None, cap=None: ROWS[:k])


def fake_ask(answer_text, cited):
    def _ask(question, conn, embed_model, top_k, llm_model, retrieval=None, max_per_paper=None):
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


# ----------------------------------------------------------------------------- multi-paper
MULTI = {"id": "M1", "type": "multi", "question": "Which tracer did both studies use?",
         "expected": "florbetapir", "doc_ids": ["a", "b"], "must_include": [["florbetapir"]],
         "evidence_all": [["completed florbetapir PET"], ["utilized Florbetapir (AV45)"]]}
MULTI_ROWS = [
    ("a:0006", "A", "2023", "Methods", 2, 2, "All participants completed florbetapir PET.", 0.80),
    ("x:0001", "X", "2020", "Intro", 1, 1, "Unrelated text.", 0.79),
    ("b:0007", "B", "2025", "Methods", 3, 3, "we utilized Florbetapir (AV45) for ADNI.", 0.78),
]


@pytest.fixture
def multi_search(monkeypatch):
    import search
    monkeypatch.setattr(search, "search", lambda q, model, conn, k, mode=None, cap=None: MULTI_ROWS[:k])


def test_multi_needs_every_part_retrieved(multi_search):
    complete = evaluate.run_one(MULTI, None, None, top_k=3, llm_model=None, retrieval_only=True)
    assert complete["hit_rank"] == 3 and complete["parts_found"] == "2/2"   # rank of the last part
    partial = evaluate.run_one(MULTI, None, None, top_k=2, llm_model=None, retrieval_only=True)
    assert partial["hit_rank"] is None and partial["parts_found"] == "1/2"


@pytest.mark.parametrize("cited, cited_gold", [([1, 3], True), ([1], False), ([2], False)])
def test_multi_must_cite_a_source_for_each_part(monkeypatch, multi_search, cited, cited_gold):
    text = "Both used florbetapir " + "".join("[{}]".format(n) for n in cited) + "."
    monkeypatch.setattr(answer, "ask", fake_ask(text, cited))
    res = evaluate.run_one(MULTI, None, None, top_k=3, llm_model="fake", retrieval_only=False)
    assert res["verdict"] == "correct" and res["cited_gold"] is cited_gold


def test_right_words_without_retrieved_evidence_are_unsupported(monkeypatch, multi_search):
    # only paper a is in the top 2: "both used florbetapir" can't be backed by the context
    monkeypatch.setattr(answer, "ask", fake_ask("Both used florbetapir [1][2].", [1, 2]))
    res = evaluate.run_one(MULTI, None, None, top_k=2, llm_model="fake", retrieval_only=False)
    assert res["verdict"] == "unsupported"
    s = evaluate.summarise([res], top_k=2, retrieval_only=False)
    assert s["multi"]["unsupported"] == 1 and s["multi"]["correct"] == 0


def test_refusal_is_not_scored_for_facts(monkeypatch, multi_search):
    # a refusal that mentions what it did find must not show "1/1 facts"
    text = answer.NOT_FOUND + " The sources mention florbetapir for ADNI only [1]."
    monkeypatch.setattr(answer, "ask", fake_ask(text, [1]))
    res = evaluate.run_one(MULTI, None, None, top_k=2, llm_model="fake", retrieval_only=False)
    assert res["verdict"] == "refused (not retrieved)" and res["facts_matched"] == "-"


def test_single_evidence_question_has_no_parts_field(fake_search):
    res = evaluate.run_one(QUESTION, None, None, top_k=2, llm_model=None, retrieval_only=True)
    assert "parts_found" not in res


# ----------------------------------------------------------------------------- --rescore
class FakeTextConn:
    """Answers rescore()'s single SELECT chunk_id, text query from a dict."""
    def __init__(self, texts):
        self.texts = texts

    def execute(self, sql, params):
        texts = self.texts

        class Result:
            def fetchall(self):
                return [(cid, texts[cid]) for cid in params[0] if cid in texts]
        return Result()


def test_rescore_applies_new_evidence_without_new_llm_calls(tmp_path, monkeypatch):
    saved = {"llm_model": "fake", "top_k": 2, "summary": {}, "results": [{
        "id": "L1", "type": "lookup", "question": "How many regions?", "expected": "29",
        "retrieved": ["paper:0003", "paper:0026"], "hit_rank": None,
        "answer": "29 regions [2].", "cited": [2], "not_found": False, "seconds_llm": 1.0,
        "verdict": "unsupported"}]}
    path = tmp_path / "eval_fake_k2.json"
    path.write_text(json.dumps(saved), encoding="utf-8")
    monkeypatch.setattr(answer, "ask", lambda *a, **k: pytest.fail("rescore must not call the LLM"))
    conn = FakeTextConn({"paper:0003": "Unrelated.", "paper:0026": "a Bonferroni correction for 29 regions"})

    out = evaluate.rescore(path, [QUESTION], conn)          # QUESTION's evidence is in paper:0026
    r = json.loads(out.read_text(encoding="utf-8"))["results"][0]
    assert out.name == "eval_fake_k2_rescored.json"
    assert r["hit_rank"] == 2 and r["verdict"] == "correct" and r["cited_gold"] is True
    assert r["answer"] == "29 regions [2]."                   # the saved answer is kept as it was


# ----------------------------------------------------------------------------- agent mode
def test_agent_results_are_scored_on_every_passage_it_saw(monkeypatch):
    import agent
    hits = [{"n": i, "chunk_id": r[0], "text": r[6]} for i, r in enumerate(MULTI_ROWS, start=1)]
    monkeypatch.setattr(agent, "ask_agent", lambda *a, **k: {
        "answer": "Both used florbetapir [1][3].", "cited": [1, 3], "not_found": False,
        "hits": hits, "llm_model": "fake", "steps": 2, "llm_calls": 3, "trace": [],
        "seconds": {"retrieval": 0.1, "llm": 1.0}})
    res = evaluate.run_one(MULTI, None, None, 5, "fake", False, mode="agent")
    assert res["verdict"] == "correct" and res["cited_gold"] is True and res["parts_found"] == "2/2"
    assert res["steps"] == 2 and res["retrieved"] == ["a:0006", "x:0001", "b:0007"]
    s = evaluate.summarise([res], 5, False, agent=True)["multi"]
    assert s["evidence found"] == 1 and "MRR" not in s and s["avg tool steps"] == 2
