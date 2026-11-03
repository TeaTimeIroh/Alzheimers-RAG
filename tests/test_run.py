"""run.py: what gets rebuilt and when, and the messages when something is missing."""
import os
import time

import pytest

import run


def touch(path, age=0.0):
    path.write_text("x", encoding="utf-8")
    t = time.time() - age
    os.utime(path, (t, t))
    return path


def test_stale_when_missing_or_older_than_an_input(tmp_path):
    src = touch(tmp_path / "in.pdf", age=10)
    out = tmp_path / "out.json"
    assert run.is_stale([out], [src])                      # missing
    touch(out, age=20)
    assert run.is_stale([out], [src])                      # older than the input
    touch(out, age=0)
    assert not run.is_stale([out], [src])
    assert run.is_stale([], [src])                         # no outputs at all


@pytest.fixture
def chunks(tmp_path, monkeypatch):
    path = tmp_path / "chunks.jsonl"
    path.write_text('{"chunk_id": "a:1"}\n{"chunk_id": "a:2"}\n', encoding="utf-8")
    monkeypatch.setattr(run, "CHUNKS", path)
    monkeypatch.setattr(run, "STATE", tmp_path / ".run_state.json")
    return path


def test_embedding_runs_until_the_database_matches_this_file(chunks):
    assert run.needs_embedding(None)                       # no table / no full-text column
    assert run.needs_embedding(2)                          # never embedded by run.py
    run.STATE.write_text('{"chunks_sha256": "%s"}' % run.file_hash(chunks), encoding="utf-8")
    assert not run.needs_embedding(2)
    assert run.needs_embedding(5)                          # table holds something else
    chunks.write_text('{"chunk_id": "a:1"}\n{"chunk_id": "b:9"}\n', encoding="utf-8")
    assert run.needs_embedding(2)                          # same count, different chunks


def test_build_skips_what_is_up_to_date(tmp_path, monkeypatch, chunks):
    pdf = touch(tmp_path / "paper.pdf", age=100)
    parsed = tmp_path / "parsed"
    parsed.mkdir()
    touch(parsed / "paper.json", age=50)
    os.utime(chunks, None)                                 # newest
    for name in ("parse_pdfs.py", "chunk_docs.py", "csv.csv"):
        touch(tmp_path / name, age=200)
    monkeypatch.setattr(run, "ROOT", tmp_path)
    monkeypatch.setattr(run, "PARSED", parsed)
    monkeypatch.setattr(run, "CSV", tmp_path / "csv.csv")
    monkeypatch.setattr(run, "database_rows", lambda: 2)
    run.STATE.write_text('{"chunks_sha256": "%s"}' % run.file_hash(chunks), encoding="utf-8")
    steps = []

    def fake_step(args):                                   # each step rewrites its output
        steps.append(args[0])
        out = {"parse_pdfs.py": parsed / "paper.json", "chunk_docs.py": chunks}.get(args[0])
        if out:
            os.utime(out, None)

    monkeypatch.setattr(run, "run_step", fake_step)

    run.build([pdf])
    assert steps == []                                     # everything current
    run.build([pdf], rebuild=True)
    assert steps == ["parse_pdfs.py", "chunk_docs.py", "embed_store.py"]
    steps.clear()
    touch(pdf, age=-5)                                     # a PDF changed: everything after it
    run.build([pdf])
    assert steps[:2] == ["parse_pdfs.py", "chunk_docs.py"]


def test_no_pdfs_explains_where_to_get_them(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "PDF_DIR", tmp_path)
    with pytest.raises(run.Stop, match="DOI links"):
        run.find_pdfs()


def test_docker_missing_or_stopped(monkeypatch):
    monkeypatch.setattr(run.shutil, "which", lambda name: None)
    with pytest.raises(run.Stop, match="isn't installed"):
        run.start_database()
    monkeypatch.setattr(run.shutil, "which", lambda name: "/usr/bin/docker")
    monkeypatch.setattr(run.subprocess, "call", lambda *a, **k: 1)       # docker info fails
    with pytest.raises(run.Stop, match="not running"):
        run.start_database()


def test_missing_packages_are_named(monkeypatch):
    monkeypatch.setattr(run.importlib.util, "find_spec", lambda name: None if name == "fitz" else 1)
    with pytest.raises(run.Stop, match="PyMuPDF"):
        run.check_packages()


def test_failed_step_stops_with_the_command(monkeypatch):
    monkeypatch.setattr(run.subprocess, "call", lambda *a, **k: 2)
    with pytest.raises(run.Stop, match="chunk_docs.py"):
        run.run_step(["chunk_docs.py"])
