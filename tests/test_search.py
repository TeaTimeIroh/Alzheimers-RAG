"""search.py: Reciprocal Rank Fusion, hybrid merging and mode selection - no database needed."""
import pytest

import search


# ----------------------------------------------------------------------------- rrf()
def test_rrf_scores_are_summed_reciprocal_ranks():
    fused = dict(search.rrf([["a", "b"], ["b", "c"]], k=60))
    assert fused["a"] == pytest.approx(1 / 61)
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused["c"] == pytest.approx(1 / 62)


def test_rrf_rewards_agreement_over_one_top_rank():
    # "b" is only 2nd in each list, but found by both: it beats "a" (1st in one list only)
    order = [cid for cid, _ in search.rrf([["a", "b"], ["c", "b"]])]
    assert order[0] == "b"


def test_rrf_ties_keep_first_seen_order():
    # a and c both score 1/61: the first list (vector search) breaks the tie
    order = [cid for cid, _ in search.rrf([["a"], ["c"]])]
    assert order == ["a", "c"]


def test_rrf_handles_empty_lists():
    assert search.rrf([[], ["x"]]) == [("x", pytest.approx(1 / 61))]
    assert search.rrf([[], []]) == []


# ----------------------------------------------------------------------------- hybrid_search()
def row(cid, score):
    return (cid, "Title " + cid, "2024", "Results", 1, 2, "text of " + cid, score)


@pytest.fixture
def fake_methods(monkeypatch):
    calls = {}

    def vec(query, model, conn, k):
        calls["vector_k"] = k
        return [row("v1", 0.9), row("both", 0.8), row("v3", 0.7)]

    def kw(query, conn, k):
        calls["keyword_k"] = k
        return [row("k1", 0.5), row("both", 0.4)]

    monkeypatch.setattr(search, "vector_search", vec)
    monkeypatch.setattr(search, "keyword_search", kw)
    return calls


def test_hybrid_takes_candidates_from_both_and_fuses(fake_methods):
    rows = search.hybrid_search("q", None, None, k=4)
    assert fake_methods == {"vector_k": search.CANDIDATES, "keyword_k": search.CANDIDATES}
    assert [r[0] for r in rows] == ["both", "v1", "k1", "v3"]
    # the row keeps its metadata and text; the score is the fused RRF score
    assert rows[0][:7] == row("both", 0)[:7]
    assert rows[0][7] == pytest.approx(1 / 62 + 1 / 62)


def test_hybrid_returns_only_top_k(fake_methods):
    assert len(search.hybrid_search("q", None, None, k=2)) == 2


# ----------------------------------------------------------------------------- search()
def test_search_dispatches_on_mode(monkeypatch):
    monkeypatch.setattr(search, "vector_search", lambda q, m, c, k: ["vector"])
    monkeypatch.setattr(search, "keyword_search", lambda q, c, k: ["keyword"])
    monkeypatch.setattr(search, "hybrid_search", lambda q, m, c, k: ["hybrid"])
    for mode in search.MODES:
        assert search.search("q", None, None, 5, mode) == [mode]


def test_default_mode_comes_from_environment(monkeypatch):
    monkeypatch.delenv("RETRIEVAL", raising=False)
    assert search.default_mode() == "vector"
    monkeypatch.setenv("RETRIEVAL", " Hybrid ")
    assert search.default_mode() == "hybrid"
    monkeypatch.setattr(search, "hybrid_search", lambda q, m, c, k: ["used hybrid"])
    assert search.search("q", None, None) == ["used hybrid"]


def test_unknown_mode_is_an_error():
    with pytest.raises(ValueError, match="retrieval mode"):
        search.search("q", None, None, 5, "bm25")
