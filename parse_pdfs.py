"""
parse_pdfs.py - Step 0 of the RAG project: turn PDFs into clean, page-aware text.

For every PDF in a folder it:
  1. extracts text block-by-block with PyMuPDF, re-ordering 2-column pages column by column,
  2. drops noise: running headers/footers, page numbers, tiny figure labels like "(a)",
     but KEEPS section headings (written as "## Results") so the chunker can use them,
  3. cleans text: fixes ligatures (ﬁ -> fi), re-joins words hyphenated across lines,
     merges the broken lines of a paragraph into one line,
  4. optionally cuts the reference list (it pollutes retrieval later),
  5. attaches title/authors/year/DOI from a PubMed CSV export (matched by DOI or title),
  6. writes  <out>/<name>.json  (machine-readable, used by the chunker in step 1)
     and     <out>/<name>.txt   (human-readable, for you to eyeball).

Works on Python 3.9+ (tested in CI on 3.9, 3.11, 3.13; Windows / Linux / macOS).

Install:   pip install -r requirements.txt
Run:       python parse_pdfs.py --pdf-dir pdfs --csv pdfs/csv-alzheimer.csv --out parsed --limit 3
"""
import argparse
import csv
import difflib
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pymupdf  # PyMuPDF (the old import name "fitz" is deprecated)

# A block whose top edge is in the top 7% (or bottom edge in the bottom 7%) of the page
# is a header/footer *candidate*. It is only dropped if it repeats on many pages or is
# just a page number - so a real paragraph near the edge survives.
MARGIN_RATIO = 0.07
MIN_BLOCK_CHARS = 25          # shorter blocks are usually figure labels / axis ticks
REFERENCE_HEADINGS = re.compile(
    r"^\s*(references|reference list|bibliography|literature cited|references and notes)\s*$",
    re.IGNORECASE,
)
# Everything after these is form/boilerplate, not paper content (cut like the reference list).
END_HEADINGS = re.compile(r"^\s*reporting summary\s*$", re.IGNORECASE)
CAPTION_RE = re.compile(r"^\s*(figure|fig\.|table|supplementary (figure|table))\s*\d", re.IGNORECASE)
CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
DOI_RE =re.compile(r"10\.\d{4,9}/[^\s\"<>]+", re.IGNORECASE)
# Section names that count as headings even if the font gives no hint.
SECTION_NAMES = re.compile(
    r"^\s*(\d+\.?\s+|[IVX]+\.\s+)?(abstract|summary|highlights|research in context|introduction|"
    r"background|methods?|materials and methods|"
    r"methods and materials|patients and methods|results( and discussion)?|discussion|conclusions?|"
    r"limitations|acknowledg(e)?ments?|funding|data availability|author contributions|"
    r"conflicts? of interest|supplementary (material|information)|study design|statistical analys[ie]s)\s*$",
    re.IGNORECASE,
)


# ----------------------------------------------------------------------------- cleaning
def normalize_text(s: str) -> str:
    """Unicode NFKC (fixes ligatures ﬁ ﬂ ﬀ, full-width chars), drop soft hyphens."""
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("­", "").replace("​", "")
    # Symbol-font glyphs without a Unicode mapping come out as control characters (\x00-\x1f):
    # brackets in formulas, ×, ±, − in p-values. Their meaning differs per font, so mark them
    # as unknown (U+FFFD) instead of guessing. (Postgres also refuses \x00 in text columns.)
    return CONTROL_CHARS.sub("�", s)


def _dehyphen(m: "re.Match") -> str:
    """Decide what to do with 'frag-<line break>next'."""
    frag, nxt = m.group(1), m.group(2)
    if nxt in ("and", "or", "to", "nor"):            # suspended hyphen: 'neuro- and glial'
        return frag + "- " + nxt
    # keep the hyphen for short or non-lowercase prefixes: p-tau, Aβ-related, APOE-stratified
    keep = len(frag) <= 2 or not frag.isalpha() or not frag.islower()
    return frag + ("-" if keep else "") + nxt        # otherwise: trans-genic -> transgenic


HYPHEN_BREAK = re.compile(r"(\w+)-\s*\n\s*([a-z]\w*)")


def join_block_lines(text: str) -> str:
    """PDF lines -> one paragraph. 'trans-\\ngenic' -> 'transgenic', other breaks -> space."""
    text = HYPHEN_BREAK.sub(_dehyphen, text)
    text = re.sub(r"\s*\n\s*", " ", text)
    return re.sub(r"[ \t]{2,}", " ", text).strip()


def continues(prev: str, nxt: str) -> bool:
    """Is `nxt` the continuation of an unfinished paragraph `prev`? (PDFs that store every
    line as its own block: double-spaced manuscripts, some journal layouts)."""
    return bool(prev) and not prev.startswith("## ") and bool(re.match(r"[a-z]", nxt)) and \
        not re.search(r"[.!?:]\)?$", prev)


def join_paragraphs(prev: str, nxt: str) -> str:
    return join_block_lines(prev + "\n" + nxt)


def signature(text: str) -> str:
    """Key used to detect repeated headers: digits removed so 'Page 3' == 'Page 7'."""
    return re.sub(r"\d+", "#", text.lower()).strip()


# ----------------------------------------------------------------------------- extraction
def _span_text(line: dict) -> str:
    return "".join(s["text"] for s in line["spans"])


def body_font_size(doc: "pymupdf.Document") -> float:
    """Most common font size, weighted by number of characters = the body-text size."""
    counts = Counter()
    for page in doc:
        for b in page.get_text("dict")["blocks"]:
            for line in b.get("lines", []):
                for s in line["spans"]:
                    counts[round(s["size"], 1)] += len(s["text"].strip())
    return counts.most_common(1)[0][0] if counts else 10.0


def order_blocks(blocks: List[dict], page_w: float) -> List[dict]:
    """Reading order. One-column page: top to bottom. Two-column page: left column, then right
    column, with full-width blocks (title, abstract, wide figures) acting as separators."""
    mid, tol = page_w / 2, page_w * 0.05
    for b in blocks:
        w = b["x1"] - b["x0"]
        if w > page_w * 0.6 or (b["x0"] < mid - tol and b["x1"] > mid + tol):
            b["col"] = "full"
        elif b["x1"] <= mid + tol:
            b["col"] = "left"
        else:
            b["col"] = "right"
    # Two-column page = both columns cover a fair share of the page height. (Counting blocks
    # fails when a whole column is a single block - then the columns were read interleaved.)
    def cover(col: str) -> float:
        return sum(b["y1"] - b["y0"] for b in blocks
                   if b["col"] == col and b["x1"] - b["x0"] > page_w * 0.3)
    page_h = max([b["y1"] for b in blocks] + [1.0])
    if cover("left") < page_h * 0.15 or cover("right") < page_h * 0.15:   # not two-column
        # Narrow blocks sitting *beside* wide text (sidebar affiliations/footnotes next to an
        # abstract) would get interleaved line by line - read them after the main text.
        def is_side(b):
            return b["x1"] - b["x0"] < page_w * 0.35 and any(
                o["x1"] - o["x0"] > page_w * 0.45 and o["y0"] < b["y1"] and b["y0"] < o["y1"]
                and (o["x0"] >= b["x1"] or o["x1"] <= b["x0"]) for o in blocks)
        side = [b for b in blocks if is_side(b)]
        main = [b for b in blocks if not is_side(b)]
        def reading_key(b):
            return (b["y0"], b["x0"])
        return sorted(main, key=reading_key) + sorted(side, key=reading_key)
    ordered, left, right = [], [], []
    def flush():
        ordered.extend(sorted(left, key=lambda b: b["y0"]))
        ordered.extend(sorted(right, key=lambda b: b["y0"]))
        left.clear()
        right.clear()
    for b in sorted(blocks, key=lambda b: (b["y0"], b["x0"])):
        if b["col"] == "full":
            flush()
            ordered.append(b)
        else:
            (left if b["col"] == "left" else right).append(b)
    flush()
    return ordered


NOT_HEADING = re.compile(
    r"^\s*(received|accepted|revised|published|corresponding|correspondence|last updated|check for|"
    r"letter\b|review article|original (paper|article|research)|podium|keywords?|key words|"
    r"electronic supplementary|copyright|\*|e-?mail|key\s*wo\s*rds)",
    re.IGNORECASE,
)


def is_heading(b: dict, body_size: float) -> bool:
    t = b["text"].replace("\n", " ").strip()
    if len(t) > 90 or b["n_lines"] > 2 or len(t.split()) > 12 or CAPTION_RE.match(t):
        return False
    if SECTION_NAMES.match(t):
        return True
    if NOT_HEADING.match(t) or len(t) < 4 or t.endswith((".", ",", ";", "-")):
        return False
    if not re.match(r"^(\d+(\.\d+)*\.?\s+)?[A-Z]", t):      # headings start with a capital (or "3.1 ")
        return False
    if t.count(",") >= 2 or re.search(r"[A-Za-z]\d+(,\d+)*(\s|,|$)", t):   # author lists "Name1,2"
        return False
    if re.search(r"\b[A-Z][A-Za-z ]{2,}:\s+\S", t):          # run-in label "METHODS: text..."
        return False
    bigger = body_size * 1.05 < b["size"] <= body_size * 1.6     # >1.6x is the paper title
    if bigger and len(t) > 60:
        return False
    if b.get("side_by_side") or b["size"] < body_size - 0.3:   # table header cells / small print
        return False
    return bigger or b["all_bold"]


def _is_bold(span: dict) -> bool:
    return bool(span["flags"] & 16) or "bold" in span["font"].lower()


def _make_block(lines: List[dict], page_h: float, body: float) -> Optional[dict]:
    """Build one block record from a list of PyMuPDF line dicts."""
    text = normalize_text("\n".join(_span_text(ln) for ln in lines)).strip()
    spans = [s for ln in lines for s in ln["spans"] if s["text"].strip()]
    if not text or not spans:
        return None
    x0 = min(ln["bbox"][0] for ln in lines)
    y0 = min(ln["bbox"][1] for ln in lines)
    x1 = max(ln["bbox"][2] for ln in lines)
    y1 = max(ln["bbox"][3] for ln in lines)
    rec = {
        "text": text, "x0": x0, "y0": y0, "x1": x1, "y1": y1,
        "in_margin": y0 < page_h * MARGIN_RATIO or y1 > page_h * (1 - MARGIN_RATIO),
        "n_lines": len(lines),
        "size": max(s["size"] for s in spans),
        "all_bold": all(_is_bold(s) for s in spans),
        # two "lines" at the same height = cells of a table row, not a wrapped heading
        # (a lone section number next to its title - "2.2" | "Generation of..." - doesn't count)
        "side_by_side": any(abs(a["bbox"][1] - b["bbox"][1]) < 2
                            and not re.fullmatch(r"\s*\d+(\.\d+)*\.?\s*", _span_text(a))
                            for a, b in zip(lines, lines[1:])),
    }
    rec["heading"] = is_heading(rec, body)
    return rec


def looks_like_prose(b: dict) -> bool:
    """Body paragraph: body-sized font and reasonably long lines (table cells are short)."""
    lines = [ln for ln in b["text"].split("\n") if ln.strip()]
    return len(b["text"]) >= 60 and sum(map(len, lines)) / max(1, len(lines)) >= 25


def _split_leading_heading(lines: List[dict], body: float) -> List[List[dict]]:
    """PyMuPDF often glues a heading to the paragraph under it ('Methods' + 30 body lines).
    If the first 1-2 lines look like a heading (bold or bigger) and the next line is plain
    body text, split them off so is_heading() can judge them on their own."""
    def style(ln):
        spans = [s for s in ln["spans"] if s["text"].strip()]
        if not spans:
            return None
        return all(_is_bold(s) for s in spans), max(s["size"] for s in spans)

    styles = [style(ln) for ln in lines]
    if len(lines) < 2 or styles[0] is None:
        return [lines]
    k = 0
    while k < min(2, len(lines) - 1) and styles[k] and (styles[k][0] or styles[k][1] > body * 1.05):
        k += 1
    if k == 0 or styles[k] is None:
        return [lines]
    nxt_bold, nxt_size = styles[k]
    if nxt_bold or abs(nxt_size - body) > 1.0:        # next line isn't body text -> no split
        return [lines]
    if k == 2 and abs(styles[0][1] - styles[1][1]) > 0.5:
        return [lines[:1], lines[1:2], lines[2:]]     # 'Methods' (big) + 'Populations' (smaller)
    return [lines[:k], lines[k:]]


def validate_headings(blocks: List[dict], body: float, first_page: bool) -> None:
    """Reject heading candidates that aren't followed by body text or another heading - this
    removes table column headers ('Name  Adverse Effects', 'Reference') - and, on page 1,
    anything before the first real section name (paper titles, author lines)."""
    seen_section = False
    # ignore short header/footer lines when looking for "the block after the heading"
    content = [b for b in blocks if not (b["in_margin"] and len(b["text"]) < 60 and not b["heading"])]
    for i, b in enumerate(content):
        if not b["heading"]:
            continue
        t = b["text"].replace("\n", " ").strip()
        if SECTION_NAMES.match(t):
            seen_section = True
            continue
        if first_page and not seen_section:
            b["heading"] = False
            continue
        nxt = content[i + 1] if i + 1 < len(content) else None
        if nxt is not None and not nxt["heading"] and not (
                abs(nxt["size"] - body) <= 1.0 and looks_like_prose(nxt)):
            b["heading"] = False


def extract_blocks(doc: "pymupdf.Document") -> List[List[dict]]:
    """Return, per page, the text blocks (in reading order) with position and font info."""
    body = body_font_size(doc)
    pages = []
    for page_no, page in enumerate(doc):
        h, w = page.rect.height, page.rect.width
        blocks = []
        for b in page.get_text("dict")["blocks"]:
            if b.get("type") != 0:               # 0 = text, 1 = image
                continue
            for part in _split_leading_heading(b["lines"], body):
                rec = _make_block(part, h, body)
                if rec:
                    blocks.append(rec)
        ordered = order_blocks(blocks, w)
        validate_headings(ordered, body, first_page=(page_no == 0))
        pages.append(ordered)
    return pages


def clean_pages(raw_pages: List[List[dict]], keep_references: bool) -> Tuple[List[dict], dict]:
    n_pages = len(raw_pages)
    # Margin blocks that appear on >= 30% of pages (min 2) are running headers/footers.
    sig_counts = Counter(
        signature(b["text"]) for blocks in raw_pages for b in blocks if b["in_margin"]
    )
    repeat_threshold = max(2, int(n_pages * 0.3))

    stats = Counter()
    out_pages = []
    in_references = in_end = False
    seen = set()
    for page_no, blocks in enumerate(raw_pages, start=1):
        paragraphs = []
        for b in blocks:
            t = b["text"]
            if b["in_margin"] and (
                sig_counts[signature(t)] >= repeat_threshold or re.fullmatch(r"[\d\s|/.-]+", t)
                # one-off footer like '1484 wileyonlinelibrary.com/journal/alz  Alzheimers Dement.'
                or (len(t) < 200 and re.search(r"www\.|https?://|\.com/|doi\.org|\bdoi:", t, re.I))
            ):
                stats["header_footer_blocks_dropped"] += 1
                continue
            if not keep_references and END_HEADINGS.match(t):
                in_end = True                    # e.g. Nature reporting summary: nothing useful after
            if in_end:
                stats["end_matter_blocks_dropped"] += 1
                continue
            if not keep_references and REFERENCE_HEADINGS.match(t):
                in_references = True
            elif in_references and b["heading"] and SECTION_NAMES.match(t.replace("\n", " ")):
                # Nature-style papers put Methods / Acknowledgements *after* the references
                in_references = False
            # Inside the reference list we still keep figure/table captions: NIH author
            # manuscripts put all figures and tables *after* the references.
            if in_references and not CAPTION_RE.match(t):
                stats["reference_blocks_dropped"] += 1
                continue
            para = join_block_lines(t)
            # the same text twice (some PDFs have a duplicated text layer)
            # (headings legitimately repeat - 'Participants' in Methods and in Results - so a
            #  heading only counts as duplicate when it repeats right after itself)
            key = para.lower()
            dup = key in seen if len(para) >= 80 else (
                b["heading"] and bool(paragraphs) and paragraphs[-1].lower() == "## " + key)
            if dup:
                stats["duplicate_blocks_dropped"] += 1
                continue
            seen.add(key)
            if b["heading"]:     # repeated running headers were already dropped above
                paragraphs.append("## " + para)
                stats["headings_kept"] += 1
                continue
            # a lowercase line continuing an unfinished paragraph is merged, even if short
            if paragraphs and continues(paragraphs[-1], para):
                paragraphs[-1] = join_paragraphs(paragraphs[-1], para)
                stats["line_blocks_merged"] += 1
                continue
            if len(t) < MIN_BLOCK_CHARS:
                stats["short_blocks_dropped"] += 1
                continue
            paragraphs.append(para)
        out_pages.append({"page": page_no, "text": "\n\n".join(paragraphs)})
    return out_pages, dict(stats)


# ----------------------------------------------------------------------------- metadata
def load_csv(csv_path: Optional[Path]) -> List[Dict[str, str]]:
    if not csv_path or not csv_path.exists():
        return []
    # utf-8-sig strips the BOM PubMed puts at the start of the file
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def squash(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", normalize_text(s).lower())


def match_metadata(doc: "pymupdf.Document", rows: List[Dict[str, str]]) -> Optional[Dict[str, str]]:
    """Find this PDF's row in the PubMed CSV: first by DOI, then by title text."""
    if not rows:
        return None
    head = normalize_text("".join(doc[i].get_text() for i in range(min(2, doc.page_count))))
    dois_in_pdf = {d.rstrip(".,;)").lower() for d in DOI_RE.findall(head)}
    for r in rows:
        if r.get("DOI") and r["DOI"].lower() in dois_in_pdf:
            return r
    # fallback: is the (squashed) CSV title contained in the squashed first-page text?
    head_sq = squash(head)
    best, best_score = None, 0.0
    for r in rows:
        title_sq = squash(r.get("Title", ""))
        if title_sq and title_sq[:60] in head_sq:
            return r
        score = difflib.SequenceMatcher(None, title_sq[:80], head_sq[:2000]).find_longest_match(
            0, len(title_sq[:80]), 0, len(head_sq[:2000])).size / max(1, len(title_sq[:80]))
        if score > best_score:
            best, best_score = r, score
    return best if best_score > 0.8 else None


# ----------------------------------------------------------------------------- main
def parse_pdf(pdf_path: Path, rows: List[Dict[str, str]], keep_references: bool) -> dict:
    doc = pymupdf.open(str(pdf_path))
    meta = match_metadata(doc, rows) or {}
    pages, stats = clean_pages(extract_blocks(doc), keep_references)
    result = {
        "doc_id": pdf_path.stem,
        "file": pdf_path.name,
        "title": meta.get("Title") or doc.metadata.get("title") or pdf_path.stem,
        "authors": meta.get("Authors", ""),
        "journal": meta.get("Journal/Book", ""),
        "year": meta.get("Publication Year", ""),
        "doi": meta.get("DOI", ""),
        "pmcid": meta.get("PMCID", ""),
        "metadata_matched": bool(meta),
        "n_pages": doc.page_count,
        # pages with (almost) no extractable text at all -> probably scanned images, need OCR
        "image_only_pages": [i + 1 for i, p in enumerate(doc) if len(p.get_text().strip()) < 50],
        "n_chars": sum(len(p["text"]) for p in pages),
        "cleaning_stats": stats,
        "pages": pages,
    }
    doc.close()
    return result


def write_outputs(result: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / (result["doc_id"] + ".json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    with open(out_dir / (result["doc_id"] + ".txt"), "w", encoding="utf-8") as f:
        f.write("TITLE:   {}\nAUTHORS: {}\nJOURNAL: {} ({})\nDOI:     {}\n".format(
            result["title"], result["authors"], result["journal"], result["year"], result["doi"]))
        f.write("CLEANING: {}\n".format(result["cleaning_stats"]))
        for p in result["pages"]:
            f.write("\n\n========== PAGE {} ==========\n\n".format(p["page"]))
            f.write(p["text"])


def main() -> None:
    ap = argparse.ArgumentParser(description="Parse PDFs into clean page-level text.")
    ap.add_argument("--pdf-dir", type=Path, required=True, help="folder with the PDFs")
    ap.add_argument("--csv", type=Path, default=None, help="PubMed CSV export (optional)")
    ap.add_argument("--out", type=Path, default=Path("parsed"), help="output folder")
    ap.add_argument("--limit", type=int, default=0, help="only parse the first N PDFs (0 = all)")
    ap.add_argument("--keep-references", action="store_true", help="don't cut reference lists")
    args = ap.parse_args()

    pdfs = sorted(args.pdf_dir.glob("*.pdf"))
    if args.limit:
        pdfs = pdfs[: args.limit]
    if not pdfs:
        sys.exit("No PDFs found in {}".format(args.pdf_dir))
    rows = load_csv(args.csv)

    print("{:<36} {:>5} {:>8} {:>5}  {}".format("file", "pages", "chars", "meta", "title"))
    for pdf in pdfs:
        try:
            r = parse_pdf(pdf, rows, args.keep_references)
        except Exception as e:  # one broken PDF shouldn't stop the batch
            print("{:<36} FAILED: {}".format(pdf.name, e))
            continue
        write_outputs(r, args.out)
        scanned = r["image_only_pages"]
        print("{:<36} {:>5} {:>8} {:>5}  {}{}".format(
            pdf.name[:36], r["n_pages"], r["n_chars"], "yes" if r["metadata_matched"] else "NO",
            r["title"][:55], "   <-- no text on pages {} (scanned? needs OCR)".format(scanned)
            if scanned else ""))
    print("\nWrote .json + .txt per PDF to: {}".format(args.out.resolve()))


if __name__ == "__main__":
    main()
