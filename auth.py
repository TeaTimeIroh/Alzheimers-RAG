"""
auth.py - API keys with scopes for the REST API.

Every request to /search or /ask must carry a key in the `X-API-Key` header. Each key has
scopes that say what it may do:

  search   GET /search  - retrieval only (no LLM call)
  ask      POST /ask    - retrieval + an LLM answer (the part that costs money or quota)

  401  no key, or a key that doesn't exist (or was revoked)
  403  a valid key without the scope the endpoint needs
  /health and the Swagger page (/docs) stay open.

Only a SHA-256 hash of each key is stored (in api_keys.json, kept out of git), so the file
alone doesn't give anyone access; the key itself is shown once, when it is created. A plain
hash is enough here because the keys are 256-bit random values - nothing to guess or look up
in a table; slow, salted hashes (bcrypt, argon2) are for human-chosen passwords. Hashes are
compared in constant time. Each request is logged with the key's *name*, never the key.

Manage keys:
  python auth.py create alice --scopes search,ask     prints the new key once
  python auth.py create demo  --scopes search
  python auth.py list
  python auth.py revoke demo

Settings (.env):
  API_KEYS_FILE   where the hashes live (default api_keys.json)
  API_AUTH=off    disables the check - for local experiments only; the API logs a warning
"""
import argparse
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

SCOPES = ("search", "ask")
KEY_PREFIX = "rag_"
HEADER = "X-API-Key"
NAME_RE = re.compile(r"^[\w.-]{1,40}$")

log = logging.getLogger("rag.auth")


def keys_file() -> Path:
    return Path(os.environ.get("API_KEYS_FILE", "api_keys.json"))


def auth_enabled() -> bool:
    return os.environ.get("API_AUTH", "on").strip().lower() not in ("off", "false", "0", "no")


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


# ----------------------------------------------------------------------------- the key store
def load(path: Optional[Path] = None) -> List[dict]:
    path = path or keys_file()
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))["keys"]


def save(keys: List[dict], path: Optional[Path] = None) -> None:
    path = path or keys_file()
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"keys": keys}, indent=2), encoding="utf-8")
    tmp.replace(path)                     # never leaves a half-written file behind


def create(name: str, scopes: List[str], path: Optional[Path] = None) -> str:
    """Add a key and return it - the only time the key itself exists outside its user's hands."""
    if not NAME_RE.match(name):
        raise ValueError("name: 1-40 letters, digits, '_', '.' or '-'")
    unknown = set(scopes) - set(SCOPES)
    if unknown or not scopes:
        raise ValueError("scopes must be some of {}, got {}".format(", ".join(SCOPES), scopes))
    keys = load(path)
    if any(k["name"] == name for k in keys):
        raise ValueError("a key named {!r} already exists (revoke it first)".format(name))
    key = KEY_PREFIX + secrets.token_urlsafe(32)          # 256 bits of randomness
    keys.append({"name": name, "sha256": hash_key(key), "scopes": sorted(set(scopes)),
                 "created": datetime.now().strftime("%Y-%m-%d %H:%M")})
    save(keys, path)
    return key


def revoke(name: str, path: Optional[Path] = None) -> bool:
    keys = load(path)
    kept = [k for k in keys if k["name"] != name]
    save(kept, path)
    return len(kept) < len(keys)


def identify(key: Optional[str], keys: List[dict]) -> Optional[dict]:
    """The stored entry for this key, or None. Compares against every entry in constant time,
    so the response time doesn't reveal how much of a hash matched."""
    if not key:
        return None
    digest = hash_key(key)
    found = None
    for entry in keys:
        if hmac.compare_digest(digest, entry["sha256"]):
            found = entry
    return found


# ----------------------------------------------------------------------------- FastAPI dependency
def require(scope: str):
    """FastAPI dependency: `Depends(require("ask"))` -> the caller's key name, or 401 / 403."""
    from fastapi import HTTPException, Request, Security
    from fastapi.security import APIKeyHeader

    header = APIKeyHeader(name=HEADER, auto_error=False,
                          description="API key; create one with: python auth.py create <name>")

    def dependency(request: Request, key: Optional[str] = Security(header)) -> str:
        if not auth_enabled():
            return "(auth off)"
        entry = identify(key, load())
        if entry is None:
            log.warning("rejected %s %s: %s", request.method, request.url.path,
                        "no key" if not key else "unknown key")
            raise HTTPException(401, "missing or invalid API key (header {})".format(HEADER),
                                headers={"WWW-Authenticate": "ApiKey"})
        if scope not in entry["scopes"]:
            log.warning("forbidden %s %s: key %r lacks scope %r", request.method,
                        request.url.path, entry["name"], scope)
            raise HTTPException(403, "this key lacks the {!r} scope".format(scope))
        log.info("%s %s by key %r", request.method, request.url.path, entry["name"])
        return entry["name"]

    return dependency


# ----------------------------------------------------------------------------- CLI
def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Manage the REST API's keys.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("create", help="create a key and print it once")
    c.add_argument("name")
    c.add_argument("--scopes", default="search,ask",
                   help="comma-separated, from: {} (default: both)".format(", ".join(SCOPES)))
    r = sub.add_parser("revoke", help="delete a key by name")
    r.add_argument("name")
    sub.add_parser("list", help="names, scopes and dates (the keys themselves aren't stored)")
    args = ap.parse_args(argv)

    if args.cmd == "create":
        try:
            key = create(args.name, [s.strip() for s in args.scopes.split(",") if s.strip()])
        except ValueError as e:
            sys.exit(str(e))
        print("key {!r} created - copy it now, it is not stored and can't be shown again:\n\n  {}\n"
              .format(args.name, key))
        print("use it as the {} header (in Swagger: the Authorize button).".format(HEADER))
    elif args.cmd == "revoke":
        print("revoked" if revoke(args.name) else "no key named {!r}".format(args.name))
    else:
        entries: List[Dict] = load()
        if not entries:
            print("no keys yet ({}): python auth.py create <name>".format(keys_file()))
        for k in entries:
            print("{:<20} {:<12} created {}".format(k["name"], ",".join(k["scopes"]), k["created"]))


if __name__ == "__main__":
    main()
