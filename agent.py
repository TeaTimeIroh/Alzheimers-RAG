"""
agent.py - an agentic version of answer.py for questions that need more than one search.

The fixed pipeline (answer.py) runs one search for the whole question and answers from the top
5 passages. That fails for questions about two studies: all five passages come from the paper
that matches the wording best (see the README). Here the LLM drives retrieval itself:

  1. it gets the question and the list of papers (id, title, year),
  2. it calls tools - search(query) over everything, or search_paper(paper_id, query) inside
     one paper - as many times as it needs, up to a step limit,
  3. every passage a tool returns gets a number [n] that stays fixed for the whole run,
  4. it answers with [n] citations, or with the same refusal sentence as answer.py.

Same answer rules as answer.py; every tool call is kept in a trace, so you can see why it
searched what it searched. Works with any OpenAI-compatible chat API that supports tool calls
(LM Studio with a tool-capable model, Gemini).

Run:  python agent.py --check-tools                  (does the configured LLM call tools?)
      python agent.py "Which tracer did both the Beijing study and the ADNI data use?"
      python agent.py "..." --show-trace --max-steps 6
"""
import argparse
import json
import re
import sys
import time
from typing import Dict, List, Optional, Tuple

import answer
from answer import NOT_FOUND, LLMError, chat_message, cited_numbers, resolve_model, strip_thinking

MAX_STEPS = 6            # LLM turns that may call tools; then it must answer
PASSAGES_PER_SEARCH = 5
TEXT_LIMIT = 1800        # characters of a passage shown to the model (chunks are ~220 words)

RULES = answer.SYSTEM_PROMPT.split("Rules:\n", 1)[1]   # the same answer rules as answer.py

AGENT_PROMPT = """You answer questions about Alzheimer's disease research papers. You cannot see
the papers directly: use the tools to find passages, then answer from them.

Papers in the collection (id | title | year):
{papers}

Tools:
- search(query): the most relevant passages from all papers.
- search_paper(paper_id, query): the most relevant passages from one paper (use an id above).
- cannot_answer(reason): use this, after searching, when the papers do not contain the answer.

How to work:
- Always search before answering: the paper list alone is not enough to know what a paper
  says. Every passage you receive has a number [n] that stays the same for the whole
  conversation; cite those numbers.
- If the question is about more than one study, identify each study's paper in the list and
  use search_paper for each, so every part of the answer comes from the right paper.
- If a search does not return what you need, search again with other words.
- Stop searching as soon as you have passages for every part of the question, then write the
  final answer as a normal message (no tool call).
- If after searching the passages do not contain the answer, call cannot_answer with a short
  reason instead of writing an answer.

Rules:
{rules}"""

TOOLS = [
    {"type": "function", "function": {
        "name": "search",
        "description": "Search all papers. Returns the most relevant numbered passages.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "what to look for, in plain words"}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "search_paper",
        "description": "Search inside one paper. Returns its most relevant numbered passages.",
        "parameters": {"type": "object", "properties": {
            "paper_id": {"type": "string", "description": "a paper id from the list"},
            "query": {"type": "string", "description": "what to look for, in plain words"}},
            "required": ["paper_id", "query"]}}},
    {"type": "function", "function": {
        "name": "cannot_answer",
        "description": "Finish without an answer because the papers do not contain it. Only "
                       "after searching.",
        "parameters": {"type": "object", "properties": {
            "reason": {"type": "string", "description": "one sentence: what is missing, and "
                                                        "what the papers do say if relevant"}},
            "required": ["reason"]}}},
]

SEARCH_FIRST = ("You have not searched yet. Use search or search_paper first; if the papers "
                "turn out not to contain the answer, call cannot_answer.")


def chat_with_retry(messages: List[dict], model: str, tools: Optional[List[dict]],
                    tries: int = 4) -> dict:
    """One agent turn. A run makes several LLM calls, so a rate limit (429) or an overloaded
    server (5xx) is retried here, per call, instead of restarting the whole run."""
    for attempt in range(tries):
        try:
            return chat_message(messages, model, tools=tools)
        except LLMError as e:
            if not re.search(r"\((429|500|502|503|504)\)", str(e)) or attempt == tries - 1:
                raise
            wait = 20 * (attempt + 1)
            print("    rate-limited/unavailable, retrying in {}s ...".format(wait), flush=True)
            time.sleep(wait)
    raise RuntimeError("unreachable")


def list_papers(conn) -> List[dict]:
    rows = conn.execute("""SELECT doc_id, min(title), min(year), count(*) FROM chunks
                           GROUP BY doc_id ORDER BY min(year), doc_id""").fetchall()
    return [{"id": r[0], "title": r[1], "year": r[2], "passages": r[3]} for r in rows]


class Passages:
    """Every passage the agent has seen, numbered in the order it first appeared."""

    def __init__(self):
        self.hits: List[dict] = []
        self.by_id: Dict[str, dict] = {}

    def add(self, row: tuple) -> Tuple[dict, bool]:
        cid, title, year, section, p0, p1, text, _score = row
        if cid in self.by_id:
            return self.by_id[cid], False
        hit = {"n": len(self.hits) + 1, "chunk_id": cid, "title": title, "year": year,
               "section": section, "page_start": p0, "page_end": p1, "text": text}
        self.hits.append(hit)
        self.by_id[cid] = hit
        return hit, True


def format_results(rows: List[tuple], passages: Passages) -> str:
    if not rows:
        return "No passages found."
    out = []
    for row in rows:
        hit, new = passages.add(row)
        head = "[{n}] {title} ({year}) - section: {section}, pages {page_start}-{page_end}".format(**hit)
        # a passage already shown keeps its number; don't spend context on it twice
        out.append(head + ("\n" + hit["text"][:TEXT_LIMIT] if new else "\n(already shown above)"))
    return "\n\n".join(out)


def run_tool(name: str, args: dict, conn, embed_model, paper_ids: set, retrieval: Optional[str],
             passages: Passages) -> Tuple[str, List[str]]:
    """Execute one tool call. Returns (text for the model, chunk ids returned). Mistakes in
    the call come back as text, so the model can correct itself instead of crashing the run."""
    from search import search
    if name not in ("search", "search_paper"):
        return "Error: unknown tool {!r}. Use search, search_paper or cannot_answer.".format(name), []
    query = str(args.get("query", "")).strip()
    if not query:
        return "Error: give a non-empty query.", []
    if name == "search":
        rows = search(query, embed_model, conn, PASSAGES_PER_SEARCH, retrieval)
    else:
        paper = str(args.get("paper_id", "")).strip()
        if paper not in paper_ids:
            return "Error: unknown paper_id {!r}. Use an id from the list.".format(paper), []
        rows = search(query, embed_model, conn, PASSAGES_PER_SEARCH, retrieval, doc_id=paper)
    return format_results(rows, passages), [r[0] for r in rows]


def parse_args(raw) -> dict:
    if isinstance(raw, dict):
        return raw
    try:
        value = json.loads(raw or "{}")
        return value if isinstance(value, dict) else {}
    except ValueError:
        return {}


def ask_agent(question: str, conn, embed_model, llm_model: Optional[str] = None,
              retrieval: Optional[str] = None, max_steps: int = MAX_STEPS) -> dict:
    """Run the agent. Returns the same fields as answer.ask() plus `trace`, `steps` and
    `llm_calls`; `sources` are all passages the agent saw, numbered as it saw them."""
    model = llm_model or resolve_model()
    papers = list_papers(conn)
    paper_ids = {p["id"] for p in papers}
    system = AGENT_PROMPT.format(
        papers="\n".join("{id} | {title} | {year}".format(**p) for p in papers), rules=RULES,
        not_found=NOT_FOUND)
    messages: List[dict] = [{"role": "system", "content": system},
                            {"role": "user", "content": question}]
    passages = Passages()
    trace: List[dict] = []
    t_llm = t_tools = 0.0
    llm_calls = searches = nudges = 0
    final = gave_up = None

    for step in range(max_steps + 1):
        last_turn = step == max_steps
        if last_turn:   # step limit: no more tools, answer from what was found
            messages.append({"role": "user", "content": "Search limit reached. Answer now from "
                             "the passages above, following the rules."})
        t0 = time.time()
        msg = chat_with_retry(messages, model, None if last_turn else TOOLS)
        t_llm += time.time() - t0
        llm_calls += 1
        calls = msg.get("tool_calls") or []
        if not calls:
            text = strip_thinking(msg.get("content"))
            if searches == 0 and not last_turn:
                # search-first rule, enforced in code: an answer (or refusal) written without
                # looking at a single passage is sent back
                nudges += 1
                messages += [{"role": "assistant", "content": text},
                             {"role": "user", "content": SEARCH_FIRST}]
                continue
            final = text
            break
        # echo the assistant turn back, with ids on every call (some servers omit them)
        for i, c in enumerate(calls):
            c.setdefault("id", "call_{}_{}".format(step, i))
            c.setdefault("type", "function")
        messages.append({"role": "assistant", "content": msg.get("content") or "",
                         "tool_calls": calls})
        # searches first, so "search + cannot_answer" in one turn counts the search
        for c in sorted(calls, key=lambda c: c.get("function", {}).get("name") == "cannot_answer"):
            fn = c.get("function", {})
            name, args = fn.get("name", ""), parse_args(fn.get("arguments"))
            t0 = time.time()
            if name == "cannot_answer":
                if searches:
                    gave_up = str(args.get("reason", "")).strip()
                    text, ids = "Noted.", []
                else:
                    text, ids = "Error: " + SEARCH_FIRST, []
            else:
                text, ids = run_tool(name, args, conn, embed_model, paper_ids, retrieval, passages)
                if not text.startswith("Error"):
                    searches += 1
            t_tools += time.time() - t0
            trace.append({"step": step + 1, "tool": name, "args": args, "returned": ids,
                          **({"error": text} if text.startswith("Error") else {})})
            messages.append({"role": "tool", "tool_call_id": c["id"], "content": text})
        if gave_up is not None:
            # a structured refusal: the flag comes from the tool call, not from matching text
            final = (NOT_FOUND + " " + gave_up).strip()
            break

    final = final or NOT_FOUND
    cited = cited_numbers(final, len(passages.hits))
    return {
        "question": question,
        "answer": final,
        "not_found": gave_up is not None or NOT_FOUND.lower().rstrip(".") in final.lower(),
        "gave_up": gave_up is not None,        # refused through the cannot_answer tool
        "searched": searches > 0,
        "nudges": nudges,                      # answers sent back by the search-first rule
        "cited": cited,
        "sources": [{k: v for k, v in h.items() if k != "text"} for h in passages.hits],
        "hits": passages.hits,                 # with text, for the evaluator
        "llm_model": model,
        "retrieval": "agent",
        "steps": len({t["step"] for t in trace if t["tool"] != "cannot_answer"}),
        "llm_calls": llm_calls,
        "trace": trace,
        "seconds": {"retrieval": round(t_tools, 2), "llm": round(t_llm, 2)},
        "messages": messages,
    }


# ----------------------------------------------------------------------------- tool check
CHECK_TOOL = [{"type": "function", "function": {
    "name": "add", "description": "Add two integers.",
    "parameters": {"type": "object", "properties": {
        "a": {"type": "integer"}, "b": {"type": "integer"}}, "required": ["a", "b"]}}}]


def check_tools(model: Optional[str] = None) -> bool:
    """Does the configured LLM return a proper tool call? Prints what it got."""
    model = model or resolve_model()
    msg = chat_message([{"role": "user", "content": "Use the add tool to add 17 and 25."}],
                       model, tools=CHECK_TOOL, max_tokens=200)
    calls = msg.get("tool_calls") or []
    if calls and calls[0].get("function", {}).get("name") == "add":
        args = parse_args(calls[0]["function"].get("arguments"))
        ok = args.get("a") == 17 and args.get("b") == 25
        print("{}: {} called add({})".format("OK" if ok else "CALLED, BUT WRONG ARGUMENTS", model, args))
        return ok
    print("NO TOOL CALL from {}. It replied: {!r}".format(model, (msg.get("content") or "")[:300]))
    print("The agent needs a model/server with tool-call support (LM Studio: a tool-capable "
          "model; check the server log for the request).")
    return False


# ----------------------------------------------------------------------------- CLI
def main() -> None:
    ap = argparse.ArgumentParser(description="Answer a question with a tool-using agent.")
    ap.add_argument("question", nargs="?")
    ap.add_argument("--check-tools", action="store_true", help="test tool calling and exit")
    ap.add_argument("--llm-model", default=None, help="overrides LLM_MODEL")
    ap.add_argument("--retrieval", choices=("vector", "keyword", "hybrid"), default=None,
                    help="search mode the tools use (default: RETRIEVAL from .env, else hybrid)")
    ap.add_argument("--max-steps", type=int, default=MAX_STEPS)
    ap.add_argument("--show-trace", action="store_true", help="print every tool call")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(errors="replace")
    except AttributeError:
        pass

    try:
        if args.check_tools:
            sys.exit(0 if check_tools(args.llm_model) else 1)
        if not args.question:
            ap.error("give a question (or --check-tools)")
        import psycopg
        from pgvector.psycopg import register_vector
        from sentence_transformers import SentenceTransformer
        from search import DEFAULT_MODEL, DSN
        embed_model = SentenceTransformer(DEFAULT_MODEL)
        with psycopg.connect(DSN, connect_timeout=5) as conn:
            register_vector(conn)
            result = ask_agent(args.question, conn, embed_model, args.llm_model, args.retrieval,
                               args.max_steps)
    except LLMError as e:
        sys.exit("LLM error: {}".format(e))

    if args.json:
        print(json.dumps({k: v for k, v in result.items() if k not in ("messages", "hits")},
                         ensure_ascii=False, indent=2))
        return
    if args.show_trace:
        for t in result["trace"]:
            print("  step {step}: {tool}({args}) -> {n} passages{err}".format(
                n=len(t["returned"]), err="  " + t["error"] if "error" in t else "", **t))
    print("\n" + result["answer"] + "\n")
    print("Sources (* = cited):")
    for s in result["sources"]:
        if s["n"] in result["cited"] or args.show_trace:
            print("  {}[{}] {} ({}) - {}, pages {}-{}   {}".format(
                "*" if s["n"] in result["cited"] else " ", s["n"], s["title"][:70], s["year"],
                s["section"][:40], s["page_start"], s["page_end"], s["chunk_id"]))
    print("\n{} | {} tool steps, {} LLM calls | tools {}s, LLM {}s".format(
        result["llm_model"], result["steps"], result["llm_calls"],
        result["seconds"]["retrieval"], result["seconds"]["llm"]))


if __name__ == "__main__":
    main()
