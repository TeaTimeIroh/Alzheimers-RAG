"""parse_pdfs.py: text cleaning helpers, reading order, and a full parse of a generated PDF."""
import pymupdf
import pytest

import parse_pdfs as pp

BODY = ("Alzheimer's disease is the most common cause of dementia and affects millions of older "
        "adults worldwide. This {word} paragraph exists only to test the parser, and it is long "
        "enough to wrap over several lines so that it looks like ordinary prose in a paper.")


# ----------------------------------------------------------------------------- cleaning
def test_normalize_fixes_ligatures_and_soft_hyphens():
    assert pp.normalize_text("signiﬁcant eﬀect") == "significant effect"
    assert pp.normalize_text("neuro­degeneration") == "neurodegeneration"


def test_normalize_replaces_control_characters():
    # symbol-font glyphs come out as control characters; \x00 would crash the Postgres insert
    out = pp.normalize_text("P \x00 MCI \x01 and 5 \x03 10\x048")
    assert "\x00" not in out and "\x01" not in out and "\x03" not in out
    assert out.count("�") == 4
    assert pp.normalize_text("keeps\nnewlines\tand tabs") == "keeps\nnewlines\tand tabs"


@pytest.mark.parametrize("raw, expected", [
    ("trans-\ngenic mice", "transgenic mice"),            # ordinary word: hyphen removed
    ("p-\ntau levels", "p-tau levels"),                    # short prefix: hyphen kept
    ("APOE-\nstratified analysis", "APOE-stratified analysis"),   # upper-case prefix: kept
    ("neuro-\nand glial cells", "neuro- and glial cells"),  # suspended hyphen: kept as is
    ("first line\nsecond line", "first line second line"),
])
def test_join_block_lines(raw, expected):
    assert pp.join_block_lines(raw) == expected


def test_continues_detects_paragraph_split_over_blocks():
    assert pp.continues("the levels were measured in", "plasma samples from")
    assert not pp.continues("The levels were measured.", "plasma samples from")
    assert not pp.continues("## Methods", "participants were")
    assert not pp.continues("an unfinished sentence", "Capitalised start")


def test_header_signature_ignores_page_numbers():
    assert pp.signature("Page 3 of 12") == pp.signature("Page 7 of 12")


@pytest.mark.parametrize("text, is_ref", [
    ("References", True), ("REFERENCES", True), ("Literature cited", True),
    ("References and notes", True), ("Reference", False), ("References to earlier work", False),
])
def test_reference_heading(text, is_ref):
    assert bool(pp.REFERENCE_HEADINGS.match(text)) is is_ref


@pytest.mark.parametrize("text", ["Introduction", "2. Materials and methods", "3 RESULTS",
                                  "Discussion", "Conclusions", "Research in context"])
def test_section_names(text):
    assert pp.SECTION_NAMES.match(text)


def test_is_heading_rejects_table_header_cells():
    block = {"text": "Name\nAdverse Effects", "n_lines": 2, "size": 9.0, "all_bold": True,
             "side_by_side": True}
    assert not pp.is_heading(block, body_size=9.0)


def test_is_heading_accepts_bold_subheading():
    block = {"text": "Genetic covariance results", "n_lines": 1, "size": 9.0, "all_bold": True,
             "side_by_side": False}
    assert pp.is_heading(block, body_size=9.0)


# ----------------------------------------------------------------------------- reading order
def _block(name, x0, y0, x1, y1):
    return {"text": name, "x0": x0, "y0": y0, "x1": x1, "y1": y1}


def test_two_columns_read_left_then_right_even_if_right_is_one_block():
    # Regression: a right column stored as ONE big block used to make the page look
    # single-column, interleaving the columns line by line.
    blocks = [
        _block("header", 40, 20, 555, 40),
        _block("left-1", 40, 60, 290, 300),
        _block("left-2", 40, 310, 290, 780),
        _block("right-all", 305, 60, 555, 780),
    ]
    order = [b["text"] for b in pp.order_blocks(blocks, page_w=595)]
    assert order == ["header", "left-1", "left-2", "right-all"]


def test_sidebar_is_read_after_the_main_text():
    blocks = [
        _block("abstract-1", 210, 260, 545, 280),
        _block("affiliation footnote", 46, 270, 184, 292),
        _block("abstract-2", 210, 283, 545, 300),
    ]
    order = [b["text"] for b in pp.order_blocks(blocks, page_w=595)]
    assert order == ["abstract-1", "abstract-2", "affiliation footnote"]


# ----------------------------------------------------------------------------- full parse
@pytest.fixture
def synthetic_pdf(tmp_path):
    """Two pages: running header, page numbers, title + DOI, two sections, a reference list."""
    doc = pymupdf.open()
    for i in range(2):
        page = doc.new_page(width=595, height=842)
        page.insert_text((60, 30), "Journal of Synthetic Tests 2026", fontsize=8, fontname="helv")
        page.insert_text((290, 820), str(i + 1), fontsize=8, fontname="helv")
        if i == 0:
            page.insert_text((60, 90), "A synthetic paper about parsing", fontsize=18, fontname="hebo")
            page.insert_text((60, 115), "doi:10.1234/synthetic.5678", fontsize=9, fontname="helv")
            page.insert_text((60, 160), "Introduction", fontsize=12, fontname="hebo")
            page.insert_textbox(pymupdf.Rect(60, 170, 535, 260), BODY.format(word="first"),
                                fontsize=10, fontname="helv")
            page.insert_text((60, 300), "Results", fontsize=12, fontname="hebo")
            page.insert_textbox(pymupdf.Rect(60, 310, 535, 400), BODY.format(word="second"),
                                fontsize=10, fontname="helv")
        else:
            page.insert_textbox(pymupdf.Rect(60, 80, 535, 170), BODY.format(word="third"),
                                fontsize=10, fontname="helv")
            page.insert_text((60, 210), "References", fontsize=12, fontname="hebo")
            page.insert_textbox(pymupdf.Rect(60, 220, 535, 300),
                                "1. Smith J, Doe A. A reference that must not be indexed. "
                                "J Test. 2020;1:1-10.", fontsize=10, fontname="helv")
    path = tmp_path / "synthetic.pdf"
    doc.save(str(path))
    return path


CSV_ROWS = [
    {"Title": "Some other paper", "DOI": "10.9999/other", "PMCID": "PMC2"},
    {"Title": "A synthetic paper about parsing", "Authors": "Doe A", "Journal/Book": "J Test",
     "Publication Year": "2026", "DOI": "10.1234/synthetic.5678", "PMCID": "PMC1"},
]


def test_parse_pdf_end_to_end(synthetic_pdf):
    result = pp.parse_pdf(synthetic_pdf, CSV_ROWS, keep_references=False)
    page1, page2 = result["pages"][0]["text"], result["pages"][1]["text"]

    assert result["metadata_matched"] and result["doi"] == "10.1234/synthetic.5678"
    assert result["year"] == "2026" and result["pmcid"] == "PMC1"
    assert "## Introduction" in page1 and "## Results" in page1
    assert "first paragraph" in page1 and "third paragraph" in page2
    assert "Journal of Synthetic Tests" not in page1 + page2      # running header removed
    assert "must not be indexed" not in page2                     # reference list removed
    assert result["image_only_pages"] == []


def test_parse_pdf_can_keep_references(synthetic_pdf):
    result = pp.parse_pdf(synthetic_pdf, CSV_ROWS, keep_references=True)
    assert "must not be indexed" in result["pages"][1]["text"]


def test_parse_pdf_without_csv_falls_back_to_file_name(synthetic_pdf):
    result = pp.parse_pdf(synthetic_pdf, [], keep_references=False)
    assert not result["metadata_matched"]
    assert result["doc_id"] == "synthetic"
