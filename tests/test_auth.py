"""auth.py: the key store (only hashes on disk), lookup, scopes and the CLI."""
import json

import pytest

import auth


@pytest.fixture
def store(tmp_path, monkeypatch):
    path = tmp_path / "keys.json"
    monkeypatch.setenv("API_KEYS_FILE", str(path))
    return path


def test_only_the_hash_is_stored(store):
    key = auth.create("alice", ["search", "ask"])
    text = store.read_text(encoding="utf-8")
    assert key.startswith(auth.KEY_PREFIX) and len(key) > 40
    assert key not in text and auth.hash_key(key) in text
    entry = json.loads(text)["keys"][0]
    assert entry["name"] == "alice" and entry["scopes"] == ["ask", "search"]


def test_identify_finds_the_right_key(store):
    a = auth.create("a", ["search"])
    b = auth.create("b", ["ask"])
    keys = auth.load()
    assert auth.identify(a, keys)["name"] == "a" and auth.identify(b, keys)["name"] == "b"
    assert auth.identify(a + "x", keys) is None
    assert auth.identify(None, keys) is None and auth.identify("", keys) is None


def test_keys_are_random(store):
    assert auth.create("one", ["search"]) != auth.create("two", ["search"])


@pytest.mark.parametrize("name, scopes", [
    ("bad name!", ["search"]),       # name with spaces / symbols
    ("ok", ["admin"]),               # unknown scope
    ("ok", []),                      # no scope at all
])
def test_create_rejects_bad_input(store, name, scopes):
    with pytest.raises(ValueError):
        auth.create(name, scopes)


def test_duplicate_names_are_refused_and_revoke_removes(store):
    auth.create("alice", ["search"])
    with pytest.raises(ValueError, match="already exists"):
        auth.create("alice", ["ask"])
    assert auth.revoke("alice") is True and auth.revoke("alice") is False
    assert auth.load() == []


def test_missing_store_means_no_keys(store):
    assert auth.load() == []


def test_auth_is_on_unless_switched_off(monkeypatch):
    monkeypatch.delenv("API_AUTH", raising=False)
    assert auth.auth_enabled()
    for value in ("off", "OFF", "false", "0", "no"):
        monkeypatch.setenv("API_AUTH", value)
        assert not auth.auth_enabled()


def test_cli_prints_the_key_once_and_lists_names_only(store, capsys):
    auth.main(["create", "demo", "--scopes", "search"])
    out = capsys.readouterr().out
    key = next(w for w in out.split() if w.startswith(auth.KEY_PREFIX))
    auth.main(["list"])
    listing = capsys.readouterr().out
    assert "demo" in listing and "search" in listing and key not in listing
