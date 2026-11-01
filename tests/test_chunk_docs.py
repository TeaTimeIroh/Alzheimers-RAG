"""chunk_docs.py: front matter, boilerplate and metadata filtering, size cap, page tracking."""
from collections import Counter

import pytest

import chunk_docs as cd

PROSE = ("The levels of soluble amyloid were measured in plasma samples from all participants "
         "and compared between groups using a linear model adjusted for age and sex. ")


def words(n, word="alpha"):
    """Filler text of exactly n words, in 10-word sentences."""
    tokens = [word + ("." if (i + 1) % 10 == 0 else "") for i in range(n)]
    return " ".join(tokens)


def make_doc(pages):
    return {"doc_id": "doc", "title": "A test paper", "year": "2024", "pmcid": "PMC1",
            "doi": "10.1/x", "pages": [{"page": i + 1, "text": t} for i, t in enumerate(pages)]}


def chunk(doc, target=220, max_words=320, overlap=40, keep_front=False, keep_boiler=False):
    return cd.chunk_doc(doc, target, max_words, overlap, keep_front, keep_boiler, Counter())


# ----------------------------------------------------------------------------- classifiers
def test_is_prose():
    assert cd.is_prose(PROSE * 2)
    assert not cd.is_prose("Jane Doe1, John Roe2,3, Ann Poe4, Bob Loe5, Kim Zoe6")
    assert not cd.is_prose("short sentence.")


@pytest.mark.parametrize("text", [
    "Received January 8, 2015. Accepted May 5, 2015.",
    "Keywords: Alzheimer's disease; amyloid; tau",
    "1 Department of Neurology, University of A, City; 2 Institute of B, University of C, City",
    "This is an open access article distributed under the Creative Commons Attribution License.",
    "Data used in preparation of this article were obtained from the ADNI database.",
])
def test_is_metadata_true(text):
    assert cd.is_metadata(text)


def test_is_metadata_false_for_prose_mentioning_a_university():
    text = PROSE + "Samples were stored at the University hospital biobank. " + PROSE
    assert not cd.is_metadata(text)


def test_join_text_repairs_hyphenation_across_pages():
    assert cd.join_text("the trans-", "genic mice") == "the transgenic mice"
    assert cd.join_text("the p-", "tau levels") == "the p-tau levels"
    assert cd.join_text("ended here", "next part") == "ended here next part"


# ----------------------------------------------------------------------------- chunking
def test_front_matter_and_boilerplate_are_dropped():
    doc = make_doc([
        "Jane Doe1, John Roe2\n\nAbstract " + PROSE * 2,
        "## Introduction\n\n" + PROSE * 3 + "\n\n## Acknowledgements\n\nWe thank the funders. " * 3,
    ])
    sections = [c["section"] for c in chunk(doc)]
    assert "Front matter" not in sections
    assert "Abstract" in sections and "Introduction" in sections
    assert not any(s.lower().startswith("acknowledg") for s in sections)


def test_keep_flags_restore_front_matter_and_boilerplate():
    doc = make_doc(["Jane Doe1, John Roe2, Ann Poe3, Bob Loe4, Kim Zoe5, Lee Moe6, Sam Noe7\n\n"
                    "## Funding\n\n" + PROSE * 2])
    sections = {c["section"] for c in chunk(doc, keep_front=True, keep_boiler=True)}
    assert {"Front matter", "Funding"} <= sections


def test_heading_less_paper_is_not_thrown_away():
    # Regression: papers with no headings before the Discussion (Nature letters, preprints)
    # used to lose their whole introduction and results as "front matter".
    doc = make_doc(["Author One1, Author Two2, Author Three3, Author Four4, Author Five5",
                    PROSE * 3, PROSE * 3 + "\n\n## Discussion\n\n" + PROSE * 2])
    chunks = chunk(doc)
    assert any(c["section"] == "Main text" for c in chunks)
    assert sum(c["n_words"] for c in chunks) > 150


def test_hard_cap_holds_including_overlap():
    # paragraphs of ~300 words: 40 words of overlap + one paragraph would exceed 320
    doc = make_doc(["## Results\n\n" + "\n\n".join(words(300, w) for w in "abcdefgh")])
    chunks = chunk(doc, target=220, max_words=320, overlap=40)
    assert len(chunks) > 1
    assert max(c["n_words"] for c in chunks) <= 320


def test_chunks_never_mix_sections():
    doc = make_doc(["## Methods\n\n" + words(200, "method") + "\n\n## Results\n\n" + words(200, "result")])
    for c in chunk(doc):
        body = c["text"]
        assert not ("method" in body and "result" in body), c["section"]


def test_paragraph_continuing_on_next_page_is_rejoined_and_spans_both_pages():
    doc = make_doc(["## Results\n\n" + PROSE + "and the effect was larger in the trans-",
                    "genic group than in controls, as expected. " + PROSE])
    chunks = chunk(doc)
    assert "transgenic group" in chunks[0]["text"]
    assert (chunks[0]["page_start"], chunks[0]["page_end"]) == (1, 2)


def test_chunk_fields_and_ids():
    chunks = chunk(make_doc(["## Introduction\n\n" + PROSE * 3]))
    c = chunks[0]
    assert c["chunk_id"] == "doc:0000"
    assert c["embed_text"].startswith("A test paper | Introduction\n")
    assert {"doi", "pmcid", "year", "page_start", "page_end", "n_words"} <= set(c)
