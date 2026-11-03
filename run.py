"""
run.py - set everything up and start the API with one command.

  python run.py            check -> start the database -> build the index if needed -> serve
  python run.py check      only report what is ready and what is missing
  python run.py build      check + build the index, don't start the API
  python run.py serve      start the API (expects a built index)
  python run.py --rebuild  parse, chunk and embed again even if nothing changed
  python run.py --port 8080

What it does, in order (each step says what to do if it can't continue):
  1. Python packages installed?                    -> pip install -r requirements.txt
  2. .env present? (created from .env.example)     LLM settings for /ask
  3. Docker running? starts the database (docker compose up -d) and waits for it
  4. PDFs in pdfs/?                                they are not in the repository (copyright):
                                                   download them from the README's DOI links
  5. parse -> chunk -> embed, each step skipped when its output is newer than its input
     (the embedding step: when the database already holds exactly the current chunks.jsonl)
  6. LLM reachable? (only /ask needs it; /search works without)
  7. API keys: creates one the first time and prints it (unless API_AUTH=off)
  8. uvicorn api:app on http://127.0.0.1:8000 - open /docs

It only runs the same scripts as the manual steps in the README, so either way gives the same
index. Works on Windows, macOS and Linux.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, List, Optional

ROOT = Path(__file__).resolve().parent
PDF_DIR = ROOT / "pdfs"
CSV = PDF_DIR / "csv-alzheimer.csv"
PARSED = ROOT / "parsed"
CHUNKS = ROOT / "chunks.jsonl"
STATE = ROOT / ".run_state.json"       # remembers which chunks.jsonl was embedded (git-ignored)
PACKAGES = {"fitz": "PyMuPDF", "psycopg": "psycopg", "pgvector": "pgvector",
            "sentence_transformers": "sentence-transformers", "fastapi": "fastapi",
            "uvicorn": "uvicorn", "requests": "requests"}


class Stop(Exception):
    """A step can't continue; the message says what to do."""


def say(status: str, text: str) -> None:
    print("  [{:<4}] {}".format(status, text), flush=True)


# ----------------------------------------------------------------------------- small helpers
def newest(paths: Iterable[Path]) -> float:
    return max((p.stat().st_mtime for p in paths), default=0.0)


def oldest(paths: Iterable[Path]) -> float:
    return min((p.stat().st_mtime for p in paths), default=0.0)


def is_stale(outputs: List[Path], inputs: List[Path]) -> bool:
    """True when an output is missing or older than the newest input."""
    if not outputs or not all(p.exists() for p in outputs):
        return True
    return oldest(outputs) < newest(inputs)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def count_lines(path: Path) -> int:
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def run_step(args: List[str]) -> None:
    """Run one of the pipeline scripts with this Python; its own output goes straight through."""
    print("\n  $ python " + " ".join(args), flush=True)
    if subprocess.call([sys.executable] + args, cwd=ROOT) != 0:
        raise Stop("`python {}` failed - see its output above.".format(" ".join(args)))


# ----------------------------------------------------------------------------- checks
def check_packages() -> None:
    missing = [pip for mod, pip in PACKAGES.items() if importlib.util.find_spec(mod) is None]
    if missing:
        raise Stop("missing packages: {}\n         install: pip install -r requirements.txt"
                   .format(", ".join(missing)))
    say("ok", "Python packages installed")


def check_env() -> None:
    env, example = ROOT / ".env", ROOT / ".env.example"
    if env.exists():
        say("ok", ".env found")
    elif example.exists():
        shutil.copy(example, env)
        say("new", ".env created from .env.example - edit it to choose LM Studio or Gemini "
                   "(only /ask needs an LLM)")


def start_database(wait: float = 60.0) -> None:
    if shutil.which("docker") is None:
        raise Stop("Docker isn't installed: https://www.docker.com/products/docker-desktop/")
    if subprocess.call(["docker", "info"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL):
        raise Stop("Docker is installed but not running: start Docker Desktop, then run this again.")
    if subprocess.call(["docker", "compose", "up", "-d"], cwd=ROOT,
                       stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT):
        raise Stop("`docker compose up -d` failed - run it yourself to see why "
                   "(often: port 5433 already in use).")
    import psycopg

    import answer  # noqa: F401  (loads .env first: DATABASE_URL may be set there)
    from search import DSN
    deadline = time.time() + wait
    while True:
        try:
            psycopg.connect(DSN, connect_timeout=3).close()
            break
        except psycopg.OperationalError:
            if time.time() > deadline:
                raise Stop("the database container started but doesn't accept connections at "
                           "{} (docker compose logs db)".format(DSN))
            time.sleep(2)
    say("ok", "database running (Docker, port 5433)")


def find_pdfs() -> List[Path]:
    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    if not pdfs:
        raise Stop("no PDFs in {}. They aren't in the repository (copyright stays with the "
                   "publishers):\n         download the 19 papers from the DOI links in the "
                   "README's Data section into that folder;\n         file names don't matter "
                   "(they are matched to the metadata by DOI).".format(PDF_DIR))
    say("ok", "{} PDFs in {}".format(len(pdfs), PDF_DIR.name))
    return pdfs


# ----------------------------------------------------------------------------- building the index
def database_rows() -> Optional[int]:
    import psycopg
    from search import DSN
    with psycopg.connect(DSN, connect_timeout=5) as conn:
        return index_state(conn)


def index_state(conn) -> Optional[int]:
    """Rows in the chunks table, or None if the table (or its full-text column) is missing."""
    has_tsv = conn.execute("""SELECT 1 FROM information_schema.columns
                              WHERE table_name = 'chunks' AND column_name = 'tsv'""").fetchone()
    if not has_tsv:
        return None
    return conn.execute("SELECT count(*) FROM chunks").fetchone()[0]


def embedded_hash() -> Optional[str]:
    try:
        return json.loads(STATE.read_text(encoding="utf-8")).get("chunks_sha256")
    except (OSError, ValueError):
        return None


def needs_embedding(rows: Optional[int]) -> bool:
    """Embed unless the database holds exactly the chunks.jsonl that was last embedded."""
    return rows is None or rows != count_lines(CHUNKS) or embedded_hash() != file_hash(CHUNKS)


def build(pdfs: List[Path], rebuild: bool = False) -> None:
    parsed = sorted(PARSED.glob("*.json"))
    if rebuild or len(parsed) < len(pdfs) or is_stale(parsed, pdfs + [CSV, ROOT / "parse_pdfs.py"]):
        run_step(["parse_pdfs.py", "--pdf-dir", "pdfs", "--csv", str(CSV.relative_to(ROOT)),
                  "--out", "parsed"])
        parsed = sorted(PARSED.glob("*.json"))
    else:
        say("skip", "parsing: parsed/ is up to date")

    if rebuild or is_stale([CHUNKS], parsed + [ROOT / "chunk_docs.py"]):
        run_step(["chunk_docs.py", "--parsed", "parsed", "--out", "chunks.jsonl"])
    else:
        say("skip", "chunking: chunks.jsonl is up to date")

    rows = database_rows()
    if rebuild or needs_embedding(rows):
        run_step(["embed_store.py", "--chunks", "chunks.jsonl"])
        STATE.write_text(json.dumps({"chunks_sha256": file_hash(CHUNKS)}), encoding="utf-8")
    else:
        say("skip", "embedding: the database already holds these {} chunks".format(rows))


# ----------------------------------------------------------------------------- before serving
def check_llm() -> None:
    import answer
    try:
        models = answer.list_models()
        say("ok", "LLM reachable at {} ({} models)".format(answer.LLM_BASE_URL, len(models)))
    except answer.LLMError as e:
        detail = " ".join(str(e).split())             # the server's whole message, on one line
        say("warn", "LLM not reachable at {} - /search works, /ask will return 502 until it is.\n"
                    "         {}".format(answer.LLM_BASE_URL, detail[:400]))


def check_keys() -> None:
    import auth
    if not auth.auth_enabled():
        say("warn", "API_AUTH is off: /search and /ask need no key")
        return
    if auth.load():
        say("ok", "API keys: {} (python auth.py list)".format(len(auth.load())))
        return
    key = auth.create("local", ["search", "ask"])
    say("new", "created an API key 'local' - copy it now, it isn't shown again:\n\n"
               "           {}\n\n         paste it into Authorize on the /docs page".format(key))


def serve(port: int) -> None:
    print("\n  API on http://127.0.0.1:{0}  -  open http://127.0.0.1:{0}/docs   (Ctrl+C stops it;"
          " the database keeps running: docker compose stop)\n".format(port), flush=True)
    try:
        subprocess.call([sys.executable, "-m", "uvicorn", "api:app", "--host", "127.0.0.1",
                         "--port", str(port)], cwd=ROOT)
    except KeyboardInterrupt:
        pass


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Set up the RAG system and start the API.")
    ap.add_argument("command", nargs="?", default="all", choices=("all", "check", "build", "serve"))
    ap.add_argument("--rebuild", action="store_true", help="parse, chunk and embed again")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args(argv)
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))

    print("\nAlzheimer's Papers RAG - {}\n".format(args.command))
    try:
        check_packages()
        check_env()
        start_database()
        if args.command == "check":
            find_pdfs()
            rows = database_rows()
            say("ok" if rows else "todo", "index: {}".format(
                "{} chunks in the database".format(rows) if rows else "not built yet"))
            check_llm()
            return 0
        if args.command in ("all", "build"):
            build(find_pdfs(), args.rebuild)
            say("ok", "index ready")
        if args.command == "serve" and not database_rows():
            raise Stop("the index isn't built yet: python run.py build")
        if args.command in ("all", "serve"):
            check_llm()
            check_keys()
            serve(args.port)
    except Stop as e:
        print("\n  [stop] {}\n".format(e))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
