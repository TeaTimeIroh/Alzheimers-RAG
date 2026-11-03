"""
judge.py - a second opinion on saved evaluation results, from a separate (stronger) LLM.

evaluate.py scores answers with keyword/regex rules. That is fast and repeatable, but it can't
tell a reworded refusal from an answer, a reversed comparison from a right one, or a claim the
cited passages don't actually make. Here a judge model reads, for every saved answer:
the question, the reference answer, the system's answer and the text of the passages it cited,
and returns

  verdict    correct / partial / wrong / refused  (unanswerable questions: correct = refused)
  supported  is every claim in the answer stated in the passages it cites? (null: no claims)
  refusal    did the answer decline, in any wording?
  reason     one sentence

It runs after the evaluation, on the JSON files in results/ - no new answers are generated,
so the same judge grades every model and mode. The judge is configured separately from the
model being evaluated (in .env):

  JUDGE_BASE_URL   default: Gemini's OpenAI-compatible endpoint
  JUDGE_MODEL      e.g. models/gemini-3.8-flash  (python answer.py --list-models shows the ids)
  JUDGE_API_KEY    your AI Studio key

Free-tier friendly: several answers are graded per call (--batch, default 4: one 38-question
run is ~10 calls), each batch is saved as soon as it is graded, and a daily-quota error stops
the run cleanly. Running the same command again later continues where it stopped.

Run:  python judge.py results/eval_..._agent_....json [more files]
      python judge.py results/....json --only M1,M8,U6      (re-grades just these)
Writes <name>_judged_<judge model>.json next to each file (one per judge, so judges can be
compared) and prints where judge and rules disagree -
those are the answers to read by hand before trusting either.

Caveats: the judge is an LLM too and can be wrong; it is checked against hand labels, not
assumed right. Grading several answers in one call saves quota but means they share a prompt.
"""
import argparse
import json
import os
import re
import sys
import time
from collections import Counter, OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

import requests

import answer                       # loads .env

JUDGE_BASE_URL = os.environ.get("JUDGE_BASE_URL",
                                "https://generativelanguage.googleapis.com/v1beta/openai").rstrip("/")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "")
JUDGE_API_KEY = os.environ.get("JUDGE_API_KEY", "")
JUDGE_TIMEOUT = float(os.environ.get("JUDGE_TIMEOUT", "300"))
# How long the judge "thinks" before replying (OpenAI-style reasoning_effort, which Gemini's
# endpoint accepts). Thinking models default to a lot of it, which made a 4-answer batch take
# minutes; "low" is plenty for grading against a reference. Empty: don't send the setting.
JUDGE_REASONING = os.environ.get("JUDGE_REASONING", "low").strip()

VERDICTS = ("correct", "partial", "wrong", "refused")
PASSAGE_CHARS = 2500                # cap per cited passage, to keep a batch well inside TPM limits

JUDGE_PROMPT = """You grade answers produced by a question-answering system over a fixed set of \
Alzheimer's disease research papers. The system must answer only from the passages it retrieved \
and cite them as [n]; if the papers don't contain the answer it should say so.

You get several items. Grade each one on its own - items are unrelated, never let one influence \
another. Each item has: id, question type, the question, a reference answer written by the \
evaluator, the system's answer, and the text of the passages the answer cites.

For each item decide:
- refusal: true if the answer declines to answer or says the papers don't contain it, in any \
wording - even if it also mentions related things it did find.
- verdict:
  * for type "unanswerable" (the reference says it is not in the papers): "correct" if the answer \
is a refusal, "wrong" if it gives an answer anyway.
  * otherwise: "correct" if every key fact of the reference answer is present and nothing \
contradicts it (other wording, order or extra correct detail is fine); "partial" if only some \
key facts are present, or the answer hedges between right and wrong; "wrong" if a key fact is \
wrong, a comparison is reversed, or a finding is attributed to the wrong study; "refused" if it \
is a refusal.
- supported: true if every factual claim in the answer is stated in the cited passages \
(judge only against the passage text shown, not your own knowledge); false if any claim is not, \
including claims with no citation; null if the answer makes no factual claims.
- reason: one short sentence explaining the verdict (and what is unsupported, if anything).

Reply with JSON only, no other text:
{"judgments": [{"id": "...", "refusal": true|false, "verdict": "...", "supported": true|false|null, \
"reason": "..."}]}"""


class JudgeError(Exception):
    pass


class QuotaExhausted(JudgeError):
    """The daily request quota is used up: stop and continue another day."""


class JudgeUnavailable(JudgeError):
    """The model keeps answering 503 "high demand": stop instead of trying every later batch."""


class JudgeTimeout(JudgeError):
    """No reply in time. Not retried automatically: the request may still count against
    the daily quota, so a hanging model would quietly use it up."""


# ----------------------------------------------------------------------------- the LLM call
def judge_chat(messages: List[dict], tries: int = 5) -> str:
    """One chat call to the judge model, with waits on per-minute rate limits / overloads."""
    if not JUDGE_MODEL:
        raise JudgeError("Set JUDGE_MODEL (and JUDGE_API_KEY) in .env - see the top of judge.py.")
    payload = {"model": JUDGE_MODEL, "messages": messages, "temperature": 0.0}
    if JUDGE_REASONING:
        payload["reasoning_effort"] = JUDGE_REASONING
    headers = {"Authorization": "Bearer " + JUDGE_API_KEY, "Content-Type": "application/json"}
    for attempt in range(tries):
        t0 = time.time()
        try:
            r = requests.post(JUDGE_BASE_URL + "/chat/completions", headers=headers, json=payload,
                              timeout=JUDGE_TIMEOUT)
        except requests.Timeout:
            raise JudgeTimeout("no reply from the judge within {:.0f}s".format(JUDGE_TIMEOUT))
        except requests.ConnectionError as e:
            raise JudgeError("Cannot reach the judge at {}: {}".format(JUDGE_BASE_URL, e))
        print("    judge replied in {:.0f}s".format(time.time() - t0), flush=True)
        if r.status_code == 200:
            try:
                return r.json()["choices"][0]["message"]["content"] or ""
            except (ValueError, KeyError, IndexError, TypeError):
                raise JudgeError("Unexpected response from the judge: {}".format(r.text[:300]))
        if r.status_code == 429 and re.search(r"per ?day", r.text, re.I):
            raise QuotaExhausted(r.text[:300])
        if r.status_code in (500, 502, 503, 504) and attempt == tries - 1:
            raise JudgeUnavailable("the judge model is overloaded/unavailable ({})".format(r.status_code))
        if r.status_code not in (429, 500, 502, 503, 504) or attempt == tries - 1:
            raise JudgeError("Judge request failed ({}): {}".format(r.status_code, r.text[:500]))
        hint = re.search(r'"retryDelay":\s*"(\d+)', r.text)           # Gemini says how long
        wait = int(hint.group(1)) + 2 if hint else 20 * (attempt + 1)
        print("    judge rate-limited/unavailable ({}), retrying in {}s ...".format(
            r.status_code, wait), flush=True)
        time.sleep(wait)
    raise RuntimeError("unreachable")


def parse_judgments(content: str) -> Dict[str, dict]:
    """The judge's reply -> {id: judgment}. Accepts {"judgments": [...]} or a bare [...] list
    (models sometimes drop the wrapper), with or without ```json fences and text around it;
    drops entries with an unknown verdict (they are graded again on the next run)."""
    content = answer.strip_thinking(content)
    starts = [i for i in (content.find("{"), content.find("[")) if i >= 0]
    if not starts:
        raise JudgeError("The judge did not reply with JSON: {}".format(content[:200]))
    try:
        data, _ = json.JSONDecoder().raw_decode(content[min(starts):])   # ignores text after it
    except ValueError as e:
        raise JudgeError("Could not parse the judge's JSON ({}): {}".format(e, content[:200]))
    items = data if isinstance(data, list) else data.get("judgments", [])
    out = {}
    for j in items:
        if not isinstance(j, dict):
            continue
        verdict = str(j.get("verdict", "")).strip().lower()
        if "id" not in j or verdict not in VERDICTS:
            continue
        supported = j.get("supported")
        out[str(j["id"])] = OrderedDict(
            verdict=verdict, supported=supported if isinstance(supported, bool) else None,
            refusal=bool(j.get("refusal")), reason=str(j.get("reason", "")).strip())
    return out


# ----------------------------------------------------------------------------- building a batch
def fetch_texts(conn, ids: List[str]) -> Dict[str, str]:
    return dict(conn.execute("SELECT chunk_id, text FROM chunks WHERE chunk_id = ANY(%s)",
                             (ids,)).fetchall())


def cited_passages(r: dict, texts: Dict[str, str]) -> List[str]:
    """The passages an answer cites, as '[n] text'. [n] is the n-th passage in `retrieved`
    (pipeline: the top-k; agent: every passage it saw, in order) - the same numbers the
    answer uses."""
    out = []
    for n in sorted(set(r.get("cited") or [])):
        if 1 <= n <= len(r["retrieved"]):
            text = texts.get(r["retrieved"][n - 1], "(passage no longer in the database)")
            if len(text) > PASSAGE_CHARS:
                text = text[:PASSAGE_CHARS] + " ..."
            out.append("[{}] {}".format(n, text))
    return out


def build_messages(items: List[dict], texts: Dict[str, str]) -> List[dict]:
    blocks = []
    for r in items:
        passages = cited_passages(r, texts)
        blocks.append("\n".join([
            "=== ITEM id={} ===".format(r["id"]),
            "type: {}".format(r["type"]),
            "question: {}".format(r["question"]),
            "reference answer: {}".format(r["expected"]),
            "system answer: {}".format(r["answer"]),
            "cited passages:",
            "\n\n".join(passages) if passages else "(none)",
        ]))
    return [{"role": "system", "content": JUDGE_PROMPT},
            {"role": "user", "content": "\n\n".join(blocks)}]


# ----------------------------------------------------------------------------- one results file
def judged_path(path: Path, model: str) -> Path:
    """results/eval_x.json -> results/eval_x_judged_<model>.json (one file per judge model)."""
    if "_judged_" in path.stem:
        return path
    slug = re.sub(r"[^\w.-]+", "_", model.split("/")[-1])
    return path.with_name("{}_judged_{}.json".format(path.stem, slug))


def judge_file(path: Path, conn, chat: Callable[[List[dict]], str] = judge_chat,
               batch: int = 4, sleep: float = 13.0, only: Optional[set] = None,
               model: Optional[str] = None) -> Path:
    """Grade every answer in a results file that has no judgment from this judge model yet
    (or the ids in `only`, again). Saves after every batch, so an interrupted run resumes."""
    model = model or JUDGE_MODEL
    out = judged_path(path, model)
    data = json.load(open(out if out.exists() else path, encoding="utf-8"))
    results = [r for r in data["results"] if "answer" in r]
    if not results:
        sys.exit("{} has no answers (retrieval-only run?)".format(path.name))

    def needs(r):
        if only is not None:
            return r["id"] in only
        return (r.get("judge") or {}).get("model") != model

    todo = [r for r in results if needs(r)]
    ids = sorted({r["retrieved"][n - 1] for r in todo for n in (r.get("cited") or [])
                  if 1 <= n <= len(r["retrieved"])})
    texts = fetch_texts(conn, ids) if ids else {}
    print("{}: {} of {} answers to grade with {}".format(path.name, len(todo), len(results), model))

    stopped = None
    for i in range(0, len(todo), batch):
        items = todo[i:i + batch]
        if i and sleep:
            time.sleep(sleep)
        print("  grading {} ...".format(",".join(r["id"] for r in items)), flush=True)
        try:
            got = parse_judgments(chat(build_messages(items, texts)))
        except QuotaExhausted:
            stopped = "the judge's daily quota is used up - run the same command again after it resets"
            break
        except JudgeUnavailable as e:
            stopped = "{} - try again later; the same command continues from here".format(e)
            break
        except JudgeTimeout as e:
            stopped = ("{} - try a smaller --batch, a lower JUDGE_REASONING or a larger "
                       "JUDGE_TIMEOUT; the same command continues from here".format(e))
            break
        except JudgeError as e:
            print("    batch {} skipped: {}".format(",".join(r["id"] for r in items), e), flush=True)
            continue
        for r in items:
            if r["id"] in got:
                r["judge"] = OrderedDict(got[r["id"]], model=model)
            else:
                print("    {}: no judgment in the reply, will be retried next run".format(r["id"]))
        for r in items:
            j = r.get("judge")
            if j and j.get("model") == model:
                print("  {:<3} rules: {:<24} judge: {:<8} supported: {!s:<5} {}".format(
                    r["id"], r["verdict"], j["verdict"], j["supported"], j["reason"][:90]), flush=True)
        save(out, data, model)

    save(out, data, model)
    report(data, model)
    if stopped:
        print("\nSTOPPED: " + stopped)
    return out


def save(out: Path, data: dict, model: str) -> None:
    data["judge_summary"] = summarise(data["results"], model)
    data["judged"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ----------------------------------------------------------------------------- comparison
def rule_bucket(verdict: str) -> str:
    """The rule-based verdict on the judge's scale. "unsupported" (right words, evidence not
    retrieved) has no judge equivalent: the judge's `supported` field covers it separately."""
    if verdict.startswith("refused"):
        return "refused"
    if verdict.startswith("wrong"):
        return "wrong"
    return verdict


def agrees(r: dict) -> bool:
    rule, judge = rule_bucket(r["verdict"]), r["judge"]["verdict"]
    if r["type"] == "unanswerable":
        return (rule == "correct") == (judge == "correct")
    if rule == "unsupported":
        return judge == "correct"         # same facts; the support question is reported apart
    return rule == judge


def summarise(results: List[dict], model: str) -> Dict[str, dict]:
    """Judge verdicts per question type, how often the cited passages back the answer, and
    agreement with the keyword rules."""
    graded = [r for r in results if (r.get("judge") or {}).get("model") == model]
    summary = OrderedDict()
    for t in ("finding", "lookup", "multi", "unanswerable", "all"):
        rs = graded if t == "all" else [r for r in graded if r["type"] == t]
        if not rs:
            continue
        verdicts = Counter(r["judge"]["verdict"] for r in rs)
        claims = [r for r in rs if r["judge"]["supported"] is not None]
        s = OrderedDict(n=len(rs))
        s.update((v, verdicts.get(v, 0)) for v in VERDICTS)
        s["supported"] = "{}/{}".format(sum(r["judge"]["supported"] for r in claims), len(claims))
        s["agrees with rules"] = "{}/{}".format(sum(agrees(r) for r in rs), len(rs))
        summary[t] = s
    return summary


def report(data: dict, model: str) -> None:
    summary = data["judge_summary"]
    if not summary:
        return
    print("\njudge: {}".format(model))
    print("  {:<13} {:>3} {:>8} {:>8} {:>6} {:>8} {:>10} {:>18}".format(
        "type", "n", *VERDICTS, "supported", "agrees with rules"))
    for t, s in summary.items():
        print("  {:<13} {:>3} {:>8} {:>8} {:>6} {:>8} {:>10} {:>18}".format(
            t, s["n"], *(s[v] for v in VERDICTS), s["supported"], s["agrees with rules"]))
    rows = [r for r in data["results"] if (r.get("judge") or {}).get("model") == model]
    disagree = [r for r in rows if not agrees(r)]
    unsupported = [r for r in rows if r["judge"]["supported"] is False and agrees(r)]
    if disagree:
        print("\njudge and rules disagree (read these answers):")
        for r in disagree:
            print("  {:<3} rules: {:<24} judge: {:<8} {}".format(
                r["id"], r["verdict"], r["judge"]["verdict"], r["judge"]["reason"]))
    if unsupported:
        print("\nsame verdict, but the judge finds claims the cited passages don't make:")
        for r in unsupported:
            print("  {:<3} {:<8} {}".format(r["id"], r["judge"]["verdict"], r["judge"]["reason"]))


def main() -> None:
    ap = argparse.ArgumentParser(description="Grade saved evaluation answers with a judge LLM.")
    ap.add_argument("files", type=Path, nargs="+", metavar="RESULTS_JSON")
    ap.add_argument("--batch", type=int, default=4, help="answers per judge call (default 4)")
    ap.add_argument("--sleep", type=float, default=13.0,
                    help="seconds between calls (default 13: under a 5 requests/minute limit)")
    ap.add_argument("--only", default="", help="comma-separated ids to (re-)grade, e.g. M8,U6")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(errors="replace")
    except AttributeError:
        pass
    if not JUDGE_MODEL:
        sys.exit("Set JUDGE_MODEL and JUDGE_API_KEY in .env (see .env.example).")
    only = {x.strip() for x in args.only.split(",") if x.strip()} or None

    import psycopg
    from search import DSN
    with psycopg.connect(DSN, connect_timeout=5) as conn:
        for path in args.files:
            out = judge_file(path, conn, batch=args.batch, sleep=args.sleep, only=only)
            print("saved to {}\n".format(out))


if __name__ == "__main__":
    main()
