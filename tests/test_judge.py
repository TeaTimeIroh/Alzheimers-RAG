"""judge.py: batching, parsing, resume after a quota stop, and the comparison with the rules."""
import json

import pytest

import judge


class FakeConn:
    def __init__(self, texts):
        self.texts = texts

    def execute(self, sql, params):
        texts = self.texts

        class Result:
            def fetchall(self):
                return [(cid, texts[cid]) for cid in params[0] if cid in texts]
        return Result()


TEXTS = {"a:1": "All participants completed florbetapir PET.", "b:1": "Unrelated passage."}


def result(qid, qtype="lookup", verdict="correct", ans="Florbetapir [1].", cited=(1,)):
    return {"id": qid, "type": qtype, "question": "Which tracer?", "expected": "florbetapir",
            "retrieved": ["a:1", "b:1"], "answer": ans, "cited": list(cited), "verdict": verdict}


def write(tmp_path, results):
    path = tmp_path / "eval_fake_agent.json"
    path.write_text(json.dumps({"llm_model": "fake", "results": results}), encoding="utf-8")
    return path


def reply(*judgments):
    return "```json\n" + json.dumps({"judgments": list(judgments)}) + "\n```"


def j(qid, verdict="correct", supported=True, refusal=False):
    return {"id": qid, "verdict": verdict, "supported": supported, "refusal": refusal,
            "reason": "because"}


def test_parse_tolerates_fences_and_drops_bad_entries():
    got = judge.parse_judgments("Sure!\n" + reply(j("L1"), {"id": "L2", "verdict": "maybe"}))
    assert list(got) == ["L1"] and got["L1"]["verdict"] == "correct"
    with pytest.raises(judge.JudgeError):
        judge.parse_judgments("no json here")


def test_parse_accepts_a_bare_list():
    # seen from gemini-3.5-flash-lite: the list without the {"judgments": ...} wrapper
    content = "```json\n" + json.dumps([j("L10"), j("L11", "wrong")], indent=1) + "\n```"
    got = judge.parse_judgments(content)
    assert got["L10"]["verdict"] == "correct" and got["L11"]["verdict"] == "wrong"


def test_prompt_contains_only_the_cited_passages():
    msgs = judge.build_messages([result("L1")], TEXTS)
    user = msgs[1]["content"]
    assert "[1] All participants completed florbetapir PET." in user
    assert "Unrelated passage" not in user and "id=L1" in user


def test_batches_and_saves_judgments(tmp_path):
    path = write(tmp_path, [result("L{}".format(i)) for i in range(1, 6)])
    calls = []

    def chat(messages):
        ids = [line.split("id=")[1].split(" ")[0] for line in messages[1]["content"].splitlines()
               if line.startswith("=== ITEM")]
        calls.append(ids)
        return reply(*[j(i) for i in ids])

    out = judge.judge_file(path, FakeConn(TEXTS), chat, batch=2, sleep=0, model="judge-x")
    assert calls == [["L1", "L2"], ["L3", "L4"], ["L5"]]
    data = json.loads(out.read_text(encoding="utf-8"))
    assert out.name == "eval_fake_agent_judged_judge-x.json"
    assert all(r["judge"]["model"] == "judge-x" for r in data["results"])
    assert data["judge_summary"]["all"]["agrees with rules"] == "5/5"


def test_quota_stop_then_resume_grades_only_the_rest(tmp_path):
    path = write(tmp_path, [result("L1"), result("L2"), result("L3")])
    state = {"n": 0}

    def chat(messages):
        state["n"] += 1
        if state["n"] == 2:
            raise judge.QuotaExhausted("per day")
        return reply(j("L1"))

    judge.judge_file(path, FakeConn(TEXTS), chat, batch=1, sleep=0, model="judge-x")
    seen = []

    def chat2(messages):
        ids = [x for x in ("L1", "L2", "L3") if "id={} ".format(x) in messages[1]["content"]]
        seen.extend(ids)
        return reply(*[j(i) for i in ids])

    out = judge.judge_file(path, FakeConn(TEXTS), chat2, batch=4, sleep=0, model="judge-x")
    assert seen == ["L2", "L3"]                             # L1 was kept from the first run
    assert all("judge" in r for r in json.loads(out.read_text(encoding="utf-8"))["results"])


def test_missing_judgment_is_retried_next_time(tmp_path):
    path = write(tmp_path, [result("L1"), result("L2")])
    judge.judge_file(path, FakeConn(TEXTS), lambda m: reply(j("L1")), batch=2, sleep=0, model="x")
    data = json.loads(judge.judged_path(path, "x").read_text(encoding="utf-8"))
    assert "judge" not in data["results"][1]


@pytest.mark.parametrize("qtype, rule, judged, agree", [
    ("lookup", "correct", "correct", True),
    ("lookup", "refused (not retrieved)", "refused", True),
    ("lookup", "unsupported", "correct", True),         # support is reported separately
    ("multi", "correct", "wrong", False),                # e.g. a reversed comparison
    ("unanswerable", "wrong (answered)", "correct", False),   # a reworded refusal
    ("unanswerable", "correct", "correct", True),
])
def test_agreement_with_rules(qtype, rule, judged, agree):
    r = result("X", qtype, rule)
    r["judge"] = {"verdict": judged, "supported": None}
    assert judge.agrees(r) is agree


def test_daily_quota_is_told_apart_from_per_minute(monkeypatch):
    class Resp:
        def __init__(self, status, text):
            self.status_code, self.text = status, text

    replies = [Resp(429, '{"quotaId": "GenerateRequestsPerMinutePerProjectPerModel", '
                         '"retryDelay": "3s"}'),
               Resp(429, '{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}')]
    waits = []
    monkeypatch.setattr(judge, "JUDGE_MODEL", "m")
    monkeypatch.setattr(judge.requests, "post", lambda *a, **k: replies.pop(0))
    monkeypatch.setattr(judge.time, "sleep", waits.append)
    with pytest.raises(judge.QuotaExhausted):
        judge.judge_chat([{"role": "user", "content": "x"}])
    assert waits == [5]                                    # the per-minute limit was waited out


def test_timeout_stops_the_run_instead_of_retrying(tmp_path):
    path = write(tmp_path, [result("L1"), result("L2")])
    calls = []

    def chat(messages):
        calls.append(1)
        raise judge.JudgeTimeout("no reply")

    judge.judge_file(path, FakeConn(TEXTS), chat, batch=1, sleep=0, model="x")
    assert len(calls) == 1                     # a hanging call is not repeated, quota is kept


def test_output_file_is_per_judge_model(tmp_path):
    path = tmp_path / "eval_a.json"
    assert judge.judged_path(path, "models/gemini-3.5-flash-lite").name == \
        "eval_a_judged_gemini-3.5-flash-lite.json"
    done = tmp_path / "eval_a_judged_gemini-3.8-flash.json"
    assert judge.judged_path(done, "anything") == done        # continuing a judged file


def test_overloaded_model_stops_the_run(tmp_path, monkeypatch):
    path = write(tmp_path, [result("L1"), result("L2")])
    calls = []

    def chat(messages):
        calls.append(1)
        raise judge.JudgeUnavailable("503")

    judge.judge_file(path, FakeConn(TEXTS), chat, batch=1, sleep=0, model="x")
    assert len(calls) == 1
