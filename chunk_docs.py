"""
chunk_docs.py - Step 1 of the RAG project: parsed papers -> retrieval-sized chunks.

Reads the .json files written by parse_pdfs.py (headings are marked "## Heading") and writes
one JSON object per line to chunks.jsonl. Each chunk has:
  chunk_id, doc_id, title, year, pmcid, doi, section, page_start, page_end, n_words,
  text        - the passage itself (shown to the user as the citation)
  embed_text  - "title | section" + text (this is what you will embed; the header gives the
                embedding model context that a bare paragraph in the middle of a paper lacks)

How chunks are built:
  * front matter (title, authors, affiliations, licence) is dropped. It ends at the first
    heading OR the first real prose paragraph - some papers (Nature Letters, preprints) have
    no headings before the Discussion, so "everything before the first heading" would throw
    away their whole introduction and results,
  * boilerplate sections (Acknowledgements, Funding, Conflicts of interest, Author
    contributions, Data availability, ...) are dropped,
  * a paragraph that continues on the next page is re-joined,
  * paragraphs are never mixed across sections (a chunk belongs to exactly one heading),
  * paragraphs are packed until ~--target-words, hard cap --max-words (overlap included),
  * an over-long paragraph is split at sentence boundaries,
  * consecutive chunks of the same section share the last ~--overlap-words words,
  * tiny leftovers are merged into the previous chunk (if that stays under the cap) or dropped.

Works on Python 3.9+ (tested in CI on 3.9, 3.11, 3.13).
Run:  python chunk_docs.py --parsed parsed --out chunks.jsonl
"""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import List

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")
MIN_WORDS = 25            # chunks shorter than this get merged or dropped

# Sections that never answer a question about the science but match lots of queries
# (cohort names, institutions, drug companies...).
BOILERPLATE = re.compile(
    r"^\s*(\d+(\.\d+)*\.?\s*)?("
    r"acknowledg|funding|financial (support|disclosure)|conflicts? of interest|competing interest|"
    r"declaration of (competing )?interest|disclosure|author(s|s'|'s|’s)? contribution|"
    r"data (and code )?availability|code availability|consent|ethics (approval|statement)|"
    r"compliance with ethical|additional information|online content|"
    r"supplementary (material|information|data)|abbreviations|reporting summary)",
    re.IGNORECASE,
)
LICENCE = re.compile(r"creative commons|open access article|all rights reserved|©", re.IGNORECASE)
# Journal metadata printed under the abstract or as page-1 footnotes.
META_START = re.compile(
    r"^\W*(received|revised|accepted|published|edited by|reviewed by|correspondence|"
    r"corresponding author|citation:|copyright|specialty section|keywords?\b|abbreviations:|"
    r"funding information|data used in (the )?preparation of this article)",
    re.IGNORECASE,
)
INSTITUTION = re.compile(
    r"\b(department|university|institute|center|centre|hospital|school of|faculty|laborator|"
    r"college|foundation|clinic)\b", re.IGNORECASE)


def is_metadata(text: str) -> bool:
    """Affiliation lists, received/accepted dates, licence notes, editor lists."""
    words = n_words(text)
    if META_START.match(text) or (LICENCE.search(text) and words < 150):
        return True
    inst = len(INSTITUTION.findall(text))
    return inst >= 3 and inst / max(1, words) > 0.04     # prose mentions a university once, not 10x


def n_words(s: str) -> int:
    return len(s.split())


def is_prose(text: str) -> bool:
    """A real paragraph, as opposed to an author list / affiliation block / licence note."""
    words = text.split()
    if len(words) < 50 or LICENCE.search(text):
        return False
    commas = text.count(",") / len(words)
    with_digits = sum(1 for w in words if re.search(r"\d", w)) / len(words)
    return commas < 0.12 and with_digits < 0.15


def continues(prev: str, nxt: str) -> bool:
    """`nxt` starts lowercase and `prev` has no closing punctuation -> same paragraph."""
    return bool(re.match(r"[a-z]", nxt)) and not re.search(r"[.!?:]\)?$", prev)


def join_text(prev: str, nxt: str) -> str:
    """Join two pieces of one paragraph, fixing a word hyphenated across the break."""
    m = re.search(r"(\w+)-$", prev)
    if m and re.match(r"[a-z]", nxt) and not re.match(r"(and|or|to|nor)\b", nxt):
        frag = m.group(1)
        keep = len(frag) <= 2 or not frag.isalpha() or not frag.islower()   # p-tau, APOE-...
        return prev[:-1] + ("-" if keep else "") + nxt
    return prev + " " + nxt


def paragraphs_with_sections(doc: dict, stats: Counter) -> List[dict]:
    """Flatten pages into [{text, page_start, page_end, section}]."""
    out: List[dict] = []
    section = "Front matter"
    after_front = "Abstract"          # label for the first prose paragraph when there's no heading
    for page in doc["pages"]:
        for para in page["text"].split("\n\n"):
            para = para.strip()
            if not para:
                continue
            if para.startswith("## "):
                section = para[3:].strip()
                continue
            if is_metadata(para):
                stats["metadata_paragraphs_dropped"] += 1
                continue
            if section == "Front matter":
                if para.lower().startswith("abstract"):          # run-in "Abstract ..." paragraph
                    out.append({"text": para, "page_start": page["page"],
                                "page_end": page["page"], "section": "Abstract"})
                    section = "Main text"
                    continue
                if is_prose(para):
                    # Front matter is over. On page 1 a heading-less first paragraph is the
                    # abstract; later (e.g. preprints with pages of affiliations) it's main text.
                    section = after_front if page["page"] == 1 else "Main text"
                    out.append({"text": para, "page_start": page["page"],
                                "page_end": page["page"], "section": section})
                    section = "Main text"
                    continue
            if out and out[-1]["section"] == section and continues(out[-1]["text"], para):
                out[-1]["text"] = join_text(out[-1]["text"], para)    # paragraph runs onto next page
                out[-1]["page_end"] = page["page"]
                stats["paragraphs_rejoined"] += 1
                continue
            out.append({"text": para, "page_start": page["page"],
                        "page_end": page["page"], "section": section})
    return out


def split_long(text: str, max_words: int) -> List[str]:
    """Split one very long paragraph into pieces of <= max_words at sentence boundaries."""
    pieces, cur = [], []
    for sent in SENT_SPLIT.split(text):
        if cur and n_words(" ".join(cur + [sent])) > max_words:
            pieces.append(" ".join(cur))
            cur = []
        cur.append(sent)
    if cur:
        pieces.append(" ".join(cur))
    return pieces


def tail_words(text: str, k: int) -> str:
    """Last ~k words of text, starting at a sentence boundary when possible."""
    if k <= 0:
        return ""
    words = text.split()
    if len(words) <= k:
        return ""
    tail = " ".join(words[-k:])
    m = SENT_SPLIT.search(tail)
    return tail[m.end():] if m else tail


def chunk_doc(doc: dict, target: int, max_words: int, overlap: int,
              keep_front: bool, keep_boiler: bool, stats: Counter) -> List[dict]:
    chunks: List[dict] = []
    groups: List[dict] = []
    for p in paragraphs_with_sections(doc, stats):
        if p["section"] == "Front matter" and not keep_front:
            stats["front_matter_paragraphs_dropped"] += 1
            continue
        if BOILERPLATE.match(p["section"]) and not keep_boiler:
            stats["boilerplate_paragraphs_dropped"] += 1
            continue
        if groups and groups[-1]["section"] == p["section"]:
            groups[-1]["paras"].append(p)
        else:
            groups.append({"section": p["section"], "paras": [p]})

    def emit(section: str, texts: List[str], pages: List[int]) -> None:
        text = " ".join(texts).strip()
        if text:
            chunks.append({"section": section, "text": text,
                           "page_start": min(pages), "page_end": max(pages)})

    for g in groups:
        start_idx = len(chunks)
        units = []                                 # (text, first page, last page)
        for p in g["paras"]:
            if n_words(p["text"]) > max_words:
                # we can't tell which page each sentence is on; give every piece the full span
                units.extend((s, p["page_start"], p["page_end"]) for s in split_long(p["text"], max_words))
            else:
                units.append((p["text"], p["page_start"], p["page_end"]))
        buf: List[str] = []
        pages: List[int] = []
        for text, p0, p1 in units:
            cur = n_words(" ".join(buf))
            if buf and (cur + n_words(text) > max_words or cur >= target):
                emit(g["section"], buf, pages)
                carry = tail_words(" ".join(buf), overlap)
                if carry and n_words(carry) + n_words(text) > max_words:
                    carry = ""                     # overlap would break the cap -> skip it
                buf = [carry] if carry else []
                pages = [pages[-1]] if carry else []   # carried words come from the previous end
            buf.append(text)
            pages.extend((p0, p1))
        if buf:
            emit(g["section"], buf, pages)
        # merge a tiny final chunk of this section into the previous one, if the cap allows
        sec_chunks = chunks[start_idx:]
        if len(sec_chunks) >= 2 and n_words(sec_chunks[-1]["text"]) < MIN_WORDS and \
                n_words(sec_chunks[-2]["text"]) + n_words(sec_chunks[-1]["text"]) <= max_words:
            last = chunks.pop()
            chunks[-1]["text"] += " " + last["text"]
            chunks[-1]["page_end"] = max(chunks[-1]["page_end"], last["page_end"])
    # drop chunks that are still tiny (e.g. a lone section with one short line)
    before = len(chunks)
    chunks = [c for c in chunks if n_words(c["text"]) >= MIN_WORDS // 2]
    stats["tiny_chunks_dropped"] += before - len(chunks)

    for i, c in enumerate(chunks):
        c.update({
            "chunk_id": "{}:{:04d}".format(doc["doc_id"], i),
            "doc_id": doc["doc_id"],
            "title": doc["title"],
            "year": doc.get("year", ""),
            "pmcid": doc.get("pmcid", ""),
            "doi": doc.get("doi", ""),
            "n_words": n_words(c["text"]),
        })
        c["embed_text"] = "{} | {}\n{}".format(doc["title"], c["section"], c["text"])
    return chunks


def main() -> None:
    ap = argparse.ArgumentParser(description="Chunk parsed papers for retrieval.")
    ap.add_argument("--parsed", type=Path, default=Path("parsed"),
                    help="folder with parse_pdfs.py .json output")
    ap.add_argument("--out", type=Path, default=Path("chunks.jsonl"))
    ap.add_argument("--target-words", type=int, default=220, help="aim for chunks of about this many words")
    ap.add_argument("--max-words", type=int, default=320, help="hard cap per chunk (overlap included)")
    ap.add_argument("--overlap-words", type=int, default=40, help="words repeated between neighbours")
    ap.add_argument("--keep-front-matter", action="store_true",
                    help="keep title/author/affiliation text (dropped by default)")
    ap.add_argument("--keep-boilerplate", action="store_true",
                    help="keep Acknowledgements, Funding, Conflicts of interest, ... sections")
    # old flag, dropping front matter is now the default
    ap.add_argument("--skip-front-matter", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    files = sorted(args.parsed.glob("*.json"))
    if not files:
        sys.exit("No .json files in {}".format(args.parsed))

    all_chunks: List[dict] = []
    stats: Counter = Counter()
    print("{:<30} {:>7} {:>9} {:>9}".format("doc", "chunks", "avg_words", "sections"))
    for f in files:
        with open(f, encoding="utf-8") as fh:
            doc = json.load(fh)
        ch = chunk_doc(doc, args.target_words, args.max_words, args.overlap_words,
                       args.keep_front_matter, args.keep_boilerplate, stats)
        all_chunks.extend(ch)
        secs = len({c["section"] for c in ch})
        avg = sum(c["n_words"] for c in ch) / max(1, len(ch))
        print("{:<30} {:>7} {:>9.0f} {:>9}".format(doc["doc_id"][:30], len(ch), avg, secs))

    with open(args.out, "w", encoding="utf-8") as fh:
        for c in all_chunks:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")

    sizes = [c["n_words"] for c in all_chunks]
    print("\n{} chunks from {} papers -> {}".format(len(all_chunks), len(files), args.out))
    print("words per chunk: min {}, median {}, max {}".format(
        min(sizes), sorted(sizes)[len(sizes) // 2], max(sizes)))
    top = Counter(c["section"] for c in all_chunks).most_common(6)
    print("most common sections:", ", ".join("{} ({})".format(s[:25], n) for s, n in top))
    print("cleanup:", ", ".join("{} {}".format(k.replace("_", " "), v) for k, v in sorted(stats.items())))


if __name__ == "__main__":
    main()
