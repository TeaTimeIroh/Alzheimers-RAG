"""Shared test setup.

The tests run without a database, an LLM, the embedding model or the paper PDFs, so CI can
install only the light dependencies (requirements-dev.txt). sentence-transformers (which pulls
in PyTorch) is replaced by a tiny stub when it isn't installed.
"""
import importlib.util
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if importlib.util.find_spec("sentence_transformers") is None:
    stub = types.ModuleType("sentence_transformers")

    class SentenceTransformer:                      # noqa: D401 - minimal stand-in
        def __init__(self, name, *args, **kwargs):
            self.name = name

        def encode(self, texts, **kwargs):
            raise RuntimeError("the real embedding model is not available in tests")

    stub.SentenceTransformer = SentenceTransformer
    sys.modules["sentence_transformers"] = stub
