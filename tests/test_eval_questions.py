"""eval_questions.jsonl is data the results depend on - check it is well-formed."""
import json
import re
from pathlib import Path

import pytest

QUESTIONS_FILE = Path(__file__).resolve().parents[1] / "eval_questions.jsonl"
QUESTIONS = [json.loads(line) for line in QUESTIONS_FILE.read_text(encoding="utf-8").splitlines()
             if line.strip()]


def test_ids_are_unique_and_types_known():
    ids = [q["id"] for q in QUESTIONS]
    assert len(ids) == len(set(ids))
    assert {q["type"] for q in QUESTIONS} <= {"finding", "lookup", "multi", "unanswerable"}


@pytest.mark.parametrize("q", QUESTIONS, ids=[q["id"] for q in QUESTIONS])
def test_question_is_complete(q):
    assert q["question"].strip() and q["expected"].strip()
    if q["type"] == "unanswerable":
        assert not q["must_include"] and not q.get("evidence") and not q.get("evidence_all")
    elif q["type"] == "multi":                      # needs passages from several papers
        assert len(q["doc_ids"]) >= 2 and len(q["evidence_all"]) == len(q["doc_ids"])
        assert q["must_include"]
    else:
        assert q["doc_id"] and q["must_include"] and q["evidence"]
        for group in q["must_include"]:
            for pattern in group:
                re.compile(pattern)                    # every fact pattern is a valid regex
