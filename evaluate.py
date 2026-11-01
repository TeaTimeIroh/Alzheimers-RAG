"""
evaluate.py - Step 5 of the RAG project: measure how well the system answers known questions.

For every question in eval_questions.jsonl it runs the full pipeline (retrieve -> LLM) and scores:
  retrieval  hit@k   - was the passage containing the answer among the k retrieved chunks?
                       (and at which rank)
  answer     correct - all required facts present (keyword/regex groups in must_include)
             partial - some of them present
             refused - the model said it could not find the answer although it was retrievable
  citation   cited the right source - one of the [n] the answer cites contains the evidence
  unanswerable       - correct only if the model refuses ("I could not find this...")

Results are printed per question and per question type, and saved to results/ as JSON (with
the full answers, for manual review) so different models / settings can be compared.

The keyword scoring is a fast approximation, not a judge: an answer can contain the right
words and still be wrong (or phrase a right answer differently). Read the answers marked
partial/wrong in the saved JSON before quoting the numbers.

Run:  python evaluate.py                       (uses the LLM settings from .env)
      python evaluate.py --retrieval-only      (no LLM: just hit@k, takes seconds)
      python evaluate.py --top-k 8 --only L1,L5
      python evaluate.py --sleep 5             (pause between questions, for free-tier limits)
"""
import argparse
import json
import re
import sys
import time
import unicodedata
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import answer                                   # loads .env, LLM client, ask()

TYPES = ("finding", "lookup", "unanswerable")


# ----------------------------------------------------------------------------- matching
def norm(s: str) -> str:
    """Lower-case, unify dashes/minus/spaces, drop thousands separators ('5,108' -> '5108')."""
    s = unicodedata.normalize("NFKC", s or "").lower()
    s = re.sub(r"[‐-―−]", "-", s)            # hyphens, en/em dash, minus
    s = re.sub(r"[   ]", " ", s)              # non-breaking / thin spaces
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)
    return re.sub(r"\s+", " ", s)


def contains_evidence(text: str, evidence) -> bool:
    """`evidence` is one passage or a list of alternatives (the same fact is often stated in
    the abstract AND in the results/discussion - any of them counts)."""
    if not evidence:
        return False
    alternatives = evidence if isinstance(evidence, list) else [evidence]
    t = norm(text)
    return any(norm(e) in t for e in alternatives)


def score_answer(ans: str, groups: List[List[str]]) -> int:
    """Number of must_include groups matched (a group matches if any of its regexes does)."""
    a = norm(ans)
    return sum(1 for g in groups if any(re.search(p, a) for p in g))


def is_gold(chunk: dict, q: dict) -> bool:
    """Does this retrieved chunk contain the answer, i.e. one of the passages listed under
    `evidence`? The list is curated by hand: papers repeat a result in the abstract, results
    and discussion in different words, so a 'miss' may be a passage that isn't listed yet -
    check the retrieved chunks before believing it, and add the passage if it really answers."""
    return contains_evidence(chunk["text"], q.get("evidence"))


# ----------------------------------------------------------------------------- one question
def run_one(q: dict, conn, embed_model, top_k: int, llm_model: Optional[str],
            retrieval_only: bool) -> dict:
    from search import search

    t0 = time.time()
    rows = search(q["question"], embed_model, conn, top_k)
    retrieved = [{"n": i, "chunk_id": r[0], "score": round(float(r[7]), 3), "text": r[6]}
                 for i, r in enumerate(rows, start=1)]
    res = OrderedDict(id=q["id"], type=q["type"], question=q["question"], expected=q["expected"],
                      retrieved=[r["chunk_id"] for r in retrieved],
                      seconds_retrieval=round(time.time() - t0, 2))

    gold = [r for r in retrieved if is_gold(r, q)]
    res["hit_rank"] = gold[0]["n"] if gold else None           # None = answer not retrieved
    gold_ns = {r["n"] for r in gold}

    if retrieval_only:
        return res

    result = ask_with_retry(q["question"], conn, embed_model, top_k, llm_model)
    ans = result["answer"]
    res.update(answer=ans, cited=result["cited"], not_found=result["not_found"],
               seconds_llm=result["seconds"]["llm"], llm_model=result["llm_model"])

    if q["type"] == "unanswerable":
        res["verdict"] = "correct" if result["not_found"] else "wrong (answered)"
        return res

    groups = q["must_include"]
    matched = score_answer(ans, groups)
    res["facts_matched"] = "{}/{}".format(matched, len(groups))
    # the answer must cite a source that actually contains the answer
    res["cited_gold"] = any(n in gold_ns for n in result["cited"])
    if result["not_found"]:
        res["verdict"] = "refused" if gold else "refused (not retrieved)"
    elif matched == len(groups):
        res["verdict"] = "correct"
    elif matched > 0:
        res["verdict"] = "partial"
    else:
        res["verdict"] = "wrong"
    return res


def ask_with_retry(question, conn, embed_model, top_k, llm_model, tries: int = 4) -> dict:
    """Free tiers answer 429 (rate limit) or 503 (overloaded) now and then: wait and retry."""
    for attempt in range(tries):
        try:
            return answer.ask(question, conn, embed_model, top_k, llm_model)
        except answer.LLMError as e:
            transient = re.search(r"\((429|500|502|503|504)\)", str(e))
            if not transient or attempt == tries - 1:
                raise
            wait = 20 * (attempt + 1)
            print("    rate-limited/unavailable, retrying in {}s ...".format(wait), flush=True)
            time.sleep(wait)
    raise RuntimeError("unreachable")


# ----------------------------------------------------------------------------- summary
def summarise(results: List[dict], top_k: int, retrieval_only: bool) -> Dict[str, dict]:
    summary = OrderedDict()
    for t in TYPES + ("all answerable",):
        rs = [r for r in results if (r["type"] == t if t in TYPES else r["type"] != "unanswerable")]
        if not rs or (retrieval_only and t == "unanswerable"):   # nothing to score there
            continue
        s = OrderedDict(n=len(rs))
        if t != "unanswerable":
            s["hit@{}".format(top_k)] = sum(r["hit_rank"] is not None for r in rs)
            s["hit@1"] = sum(r["hit_rank"] == 1 for r in rs)
            ranks = [1.0 / r["hit_rank"] for r in rs if r["hit_rank"]]
            s["MRR"] = round(sum(ranks) / len(rs), 3)          # mean reciprocal rank
        if not retrieval_only:
            for v in ("correct", "partial", "wrong", "refused"):
                s[v] = sum(r["verdict"].startswith(v) for r in rs)
            if t != "unanswerable":
                s["cited right source"] = sum(bool(r.get("cited_gold")) for r in rs)
            s["avg LLM s"] = round(sum(r["seconds_llm"] for r in rs) / len(rs), 1)
        summary[t] = s
    return summary


def print_summary(summary: Dict[str, dict]) -> None:
    print("\n=== Summary ===")
    for t, s in summary.items():
        n = s["n"]
        parts = []
        for k, v in s.items():
            if k == "n":
                continue
            parts.append("{} {}".format(k, v) if isinstance(v, float) or k.startswith("avg")
                         else "{} {}/{}".format(k, v, n))
        print("  {:<15} (n={:>2})  {}".format(t, n, "  |  ".join(parts)))


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate retrieval and answers on known questions.")
    ap.add_argument("--questions", type=Path, default=Path("eval_questions.jsonl"))
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--llm-model", default=None, help="overrides LLM_MODEL from .env")
    ap.add_argument("--retrieval-only", action="store_true", help="skip the LLM, score retrieval")
    ap.add_argument("--only", default="", help="comma-separated question ids, e.g. L1,F3")
    ap.add_argument("--sleep", type=float, default=0.0, help="seconds to wait between questions")
    ap.add_argument("--out-dir", type=Path, default=Path("results"))
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(errors="replace")
    except AttributeError:
        pass

    questions = [json.loads(line) for line in open(args.questions, encoding="utf-8") if line.strip()]
    if args.only:
        wanted = {x.strip() for x in args.only.split(",")}
        questions = [q for q in questions if q["id"] in wanted]
    if not questions:
        sys.exit("no questions selected")

    import psycopg
    from pgvector.psycopg import register_vector
    from sentence_transformers import SentenceTransformer
    from search import DSN, DEFAULT_MODEL

    try:
        llm_model = None if args.retrieval_only else (args.llm_model or answer.resolve_model())
    except answer.LLMError as e:
        sys.exit("LLM error: {}".format(e))
    embed_model = SentenceTransformer(DEFAULT_MODEL)

    results = []
    print("{} questions, top-k {}, LLM: {}\n".format(
        len(questions), args.top_k, "none (retrieval only)" if args.retrieval_only else llm_model))
    with psycopg.connect(DSN) as conn:
        register_vector(conn)
        for i, q in enumerate(questions):
            if i and args.sleep:
                time.sleep(args.sleep)
            try:
                r = run_one(q, conn, embed_model, args.top_k, llm_model, args.retrieval_only)
            except answer.LLMError as e:
                sys.exit("LLM error on {}: {}".format(q["id"], e))
            results.append(r)
            rank = "-" if q["type"] == "unanswerable" else (r["hit_rank"] or "miss")
            line = "  {:<3} {:<12} retrieved@{:<4}".format(r["id"], r["type"], rank)
            if not args.retrieval_only:
                line += " {:<26} {:>5} cited-right:{:<5} {:>5.1f}s".format(
                    r["verdict"], r.get("facts_matched", ""),
                    str(r.get("cited_gold", "")) if q["type"] != "unanswerable" else "-",
                    r["seconds_llm"])
            print(line, flush=True)

    summary = summarise(results, args.top_k, args.retrieval_only)
    print_summary(summary)

    args.out_dir.mkdir(exist_ok=True)
    tag = "retrieval" if args.retrieval_only else re.sub(r"[^\w.-]+", "_", llm_model)
    out = args.out_dir / "eval_{}_k{}_{}.json".format(tag, args.top_k,
                                                      datetime.now().strftime("%Y%m%d-%H%M"))
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"llm_model": llm_model, "top_k": args.top_k, "summary": summary,
                   "results": results}, f, ensure_ascii=False, indent=2)
    print("\nfull answers saved to {}  (review the partial/wrong ones by hand)".format(out))


if __name__ == "__main__":
    main()
