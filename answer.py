"""
answer.py - Step 4 of the RAG project: retrieve the top chunks and let an LLM answer from them only.

  1. retrieves the top-k chunks for the question (same search as search.py),
  2. numbers them [1]..[k] with title / section / pages,
  3. asks the LLM to answer ONLY from those sources and cite them like [2],
  4. prints the answer plus the source list (cited sources marked with *).

Works with any OpenAI-compatible chat API - chosen with three settings, read from environment
variables or from a .env file next to this script (see .env.example):
  LLM_BASE_URL   LM Studio: http://localhost:1234/v1   (default)
                 Gemini:    https://generativelanguage.googleapis.com/v1beta/openai
  LLM_MODEL      model id; leave empty to use the first model the server lists
  LLM_API_KEY    LM Studio ignores it; Gemini needs your AI Studio key

Run:  python answer.py "Which variants' effects depend on APOE e4?"
      python answer.py "..." --top-k 8 --show-prompt
      python answer.py "..." --retrieval hybrid      (vector | keyword | hybrid, see search.py)
      python answer.py --list-models
      python answer.py "..." --json          (machine-readable, for the evaluation script)

Python 3.11+ (tested in CI on 3.11, 3.13, 3.14). Uses plain `requests` (no vendor SDK).
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import requests


# ----------------------------------------------------------------------------- settings
def load_dotenv(path: Path) -> None:
    """Minimal .env reader: KEY=VALUE lines; real environment variables win."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_dotenv(Path(__file__).with_name(".env"))
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:1234/v1").rstrip("/")
LLM_MODEL = os.environ.get("LLM_MODEL", "")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "not-needed")
LLM_TIMEOUT = float(os.environ.get("LLM_TIMEOUT", "300"))      # local models can be slow

NOT_FOUND = "I could not find this in the provided papers."

SYSTEM_PROMPT = """You answer questions about Alzheimer's disease research papers.
You are given numbered sources, which are passages from those papers.

Rules:
- Answer ONLY with information stated in the sources. Do not use outside knowledge.
- After each claim, cite the source(s) it comes from in square brackets, e.g. [2] or [1][3].
- If the sources do not contain the answer, reply exactly: "{not_found}"
- If the sources only partly answer the question, say what they do say and what is missing.
- Copy numbers, gene names, p-values and sample sizes exactly as written.
- The character � marks a symbol lost when the PDF was converted to text (often x, +/-
  or a minus sign). Do not guess it; quote the value as written or say it is unclear.
- Be concise: a few sentences, no preamble.""".format(not_found=NOT_FOUND)


# ----------------------------------------------------------------------------- LLM client
class LLMError(RuntimeError):
    pass


def _headers() -> Dict[str, str]:
    return {"Authorization": "Bearer " + LLM_API_KEY, "Content-Type": "application/json"}


def list_models() -> List[str]:
    try:
        r = requests.get(LLM_BASE_URL + "/models", headers=_headers(), timeout=15)
    except requests.ConnectionError:
        raise LLMError("Cannot reach {}. Is the LM Studio server running "
                       "(Developer tab -> Start server)?".format(LLM_BASE_URL))
    if r.status_code != 200:
        raise LLMError("GET /models failed ({}): {}".format(r.status_code, r.text[:300]))
    return [m["id"] for m in r.json().get("data", [])]


def resolve_model() -> str:
    if LLM_MODEL:
        return LLM_MODEL
    models = list_models()
    if not models:
        raise LLMError("The server lists no models. Load one in LM Studio or set LLM_MODEL.")
    return models[0]


def chat(messages: List[Dict[str, str]], model: str, temperature: float = 0.0,
         max_tokens: int = 700) -> str:
    payload = {"model": model, "messages": messages,
               "temperature": temperature, "max_tokens": max_tokens}
    try:
        r = requests.post(LLM_BASE_URL + "/chat/completions", headers=_headers(),
                          json=payload, timeout=LLM_TIMEOUT)
    except requests.ConnectionError:
        raise LLMError("Cannot reach {}. Is the LM Studio server running "
                       "(Developer tab -> Start server)?".format(LLM_BASE_URL))
    except requests.Timeout:
        raise LLMError("No answer within {:.0f}s (raise LLM_TIMEOUT for slow local models)."
                       .format(LLM_TIMEOUT))
    if r.status_code != 200:
        raise LLMError("LLM request failed ({}): {}".format(r.status_code, r.text[:500]))
    content = r.json()["choices"][0]["message"].get("content") or ""
    # Reasoning models (e.g. Qwen3 thinking variants) may include their thinking; drop it.
    return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()


# ----------------------------------------------------------------------------- RAG
def build_messages(question: str, hits: List[dict]) -> List[Dict[str, str]]:
    blocks = []
    for h in hits:
        blocks.append("[{n}] {title} ({year}) - section: {section}, pages {p0}-{p1}\n{text}".format(
            n=h["n"], title=h["title"], year=h["year"], section=h["section"],
            p0=h["page_start"], p1=h["page_end"], text=h["text"]))
    user = "Sources:\n\n{}\n\nQuestion: {}".format("\n\n".join(blocks), question)
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def cited_numbers(answer: str, n_sources: int) -> List[int]:
    """[2], [1][3], [1, 3] and [1-3] -> sorted list of valid source numbers."""
    nums = set()
    for group in re.findall(r"\[([\d,\s\-–]+)\]", answer):
        for part in re.split(r"[,\s]+", group.strip()):
            m = re.fullmatch(r"(\d+)[\-–](\d+)", part)
            if m:
                nums.update(range(int(m.group(1)), int(m.group(2)) + 1))
            elif part.isdigit():
                nums.add(int(part))
    return sorted(n for n in nums if 1 <= n <= n_sources)


def ask(question: str, conn, embed_model, top_k: int = 5, llm_model: Optional[str] = None,
        retrieval: Optional[str] = None, max_per_paper: Optional[int] = None) -> dict:
    """Retrieve, prompt, answer. Returns everything the CLI, the API and the evaluation need.
    retrieval: vector | keyword | hybrid (default: RETRIEVAL from .env, else hybrid)."""
    from search import default_mode, search      # same retrieval code as search.py

    retrieval = retrieval or default_mode()
    t0 = time.time()
    rows = search(question, embed_model, conn, top_k, retrieval, max_per_paper)
    t_retrieval = time.time() - t0
    hits = []
    for n, (cid, title, year, section, p0, p1, text, score) in enumerate(rows, start=1):
        hits.append({"n": n, "chunk_id": cid, "title": title, "year": year, "section": section,
                     "page_start": p0, "page_end": p1, "text": text, "score": float(score)})

    model = llm_model or resolve_model()
    messages = build_messages(question, hits)
    t1 = time.time()
    answer = chat(messages, model)
    t_llm = time.time() - t1

    cited = cited_numbers(answer, len(hits))
    return {
        "question": question,
        "answer": answer,
        "not_found": NOT_FOUND.lower().rstrip(".") in answer.lower(),
        "cited": cited,
        "sources": [{k: v for k, v in h.items() if k != "text"} for h in hits],
        "llm_model": model,
        "retrieval": retrieval,
        "seconds": {"retrieval": round(t_retrieval, 2), "llm": round(t_llm, 2)},
        "messages": messages,
    }


# ----------------------------------------------------------------------------- CLI
def main() -> None:
    ap = argparse.ArgumentParser(description="Answer a question from the papers, with sources.")
    ap.add_argument("question", nargs="?")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--retrieval", choices=("vector", "keyword", "hybrid"), default=None,
                    help="overrides RETRIEVAL from .env (default: hybrid)")
    ap.add_argument("--max-per-paper", type=int, default=None,
                    help="at most this many passages from one paper (default: no cap)")
    ap.add_argument("--embed-model", default=None, help="must match embed_store.py (default: BGE small)")
    ap.add_argument("--llm-model", default=None, help="overrides LLM_MODEL")
    ap.add_argument("--show-prompt", action="store_true", help="print exactly what the LLM received")
    ap.add_argument("--json", action="store_true", help="print the result as JSON")
    ap.add_argument("--list-models", action="store_true", help="list models the LLM server offers")
    args = ap.parse_args()

    try:  # Windows consoles: never crash on β, ε, ≥ ...
        sys.stdout.reconfigure(errors="replace")
    except AttributeError:
        pass

    try:
        if args.list_models:
            print("\n".join(list_models()))
            return
        if not args.question:
            ap.error("give a question (or --list-models)")

        import psycopg
        from pgvector.psycopg import register_vector
        from sentence_transformers import SentenceTransformer
        from search import DSN, DEFAULT_MODEL

        embed_model = SentenceTransformer(args.embed_model or DEFAULT_MODEL)
        with psycopg.connect(DSN) as conn:
            register_vector(conn)
            result = ask(args.question, conn, embed_model, args.top_k, args.llm_model, args.retrieval,
                         args.max_per_paper)
    except LLMError as e:
        sys.exit("LLM error: {}".format(e))

    if args.json:
        out = dict(result)
        if not args.show_prompt:
            out.pop("messages")
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return
    if args.show_prompt:
        for m in result["messages"]:
            print("----- {} -----\n{}\n".format(m["role"].upper(), m["content"]))

    print("\n" + result["answer"] + "\n")
    print("Sources (* = cited in the answer):")
    for s in result["sources"]:
        line = "  {mark}[{n}] {title} ({year})\n       {section}, pages {p0}-{p1}   {cid}   score {score:.3f}"
        print(line.format(
            mark="*" if s["n"] in result["cited"] else " ", n=s["n"], title=s["title"][:80],
            year=s["year"], section=s["section"][:50], p0=s["page_start"], p1=s["page_end"],
            cid=s["chunk_id"], score=s["score"]))
    if not result["cited"] and not result["not_found"]:
        print("\n  warning: the answer cites no source - treat it as unsupported.")
    print("\n{}  |  {} retrieval {}s, LLM {}s".format(
        result["llm_model"], result["retrieval"], result["seconds"]["retrieval"],
        result["seconds"]["llm"]))


if __name__ == "__main__":
    main()
