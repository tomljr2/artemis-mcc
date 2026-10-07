"""Download NASA Apollo history books from NTRS and turn them into plain training text.

Each book is a scanned PDF with a built-in text layer. We keep the pages that read as
prose, skip the photo pages (which come out as scanner noise), and clean up what the PDF
format does to text: words split across lines, ligatures, curly quotes.

Only books with an `include` row in data/license_log.csv are processed.

Run:  python -m data.prepare_nasa_books
"""

import csv
import re
import shutil
import ssl
import unicodedata
import urllib.request
from pathlib import Path

import certifi
from pypdf import PdfReader

# (license-log id, NTRS id). The NTRS id is the number in the document's NTRS address.
BOOKS = [
    ("sp-350", "19760005868"),  # Apollo Expeditions to the Moon (1975)
    ("sp-4204", "19790003956"),  # Moonport (1978)
    ("sp-4205", "19790020032"),  # Chariots for Apollo (1979)
    ("sp-4214", "19890016575"),  # Where No Man Has Gone Before (1988)
]
LICENSE_LOG = Path("data/license_log.csv")
RAW_DIR = Path("data/raw/nasa_books")
TEXT_DIR = Path("data/processed/nasa_books")

# Curly quotes and dashes -> their plain-ASCII equivalents. Written as escape codes because
# on screen they look almost the same as the plain characters they become.
PLAIN = {"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u2013": "-", "\u2014": "-"}


def pdf_url(ntrs_id: str) -> str:
    return f"https://ntrs.nasa.gov/api/citations/{ntrs_id}/downloads/{ntrs_id}.pdf"


def is_prose(text: str) -> bool:
    """True for a page of book text, False for photo pages and short captions.

    Scanner noise is mostly symbols and stray capitals; prose is mostly ordinary words.
    So: enough text, and most "words" look like real ones (3+ letters, lowercase after
    the first letter).
    """
    words = re.findall(r"[A-Za-z]+", text)
    if len(text) < 800 or not words:
        return False
    real = [w for w in words if len(w) >= 3 and w[1:].islower()]
    return len(real) / len(words) > 0.6


def clean_page(text: str) -> str:
    """One page of PDF text -> one paragraph of plain ASCII."""
    text = unicodedata.normalize("NFKC", text)  # e.g. the one-glyph "fi" ligature -> "fi"
    text = "".join(PLAIN.get(ch, ch) for ch in text)
    text = re.sub(r"-\n(?=[a-z])", "", text)  # "remark-\nably" -> "remarkably"
    text = re.sub(r"\s+", " ", text)  # line breaks, tabs, runs of spaces -> one space
    # Drop anything left outside printable ASCII, so stray symbols don't enter the vocabulary.
    return "".join(ch for ch in text if " " <= ch <= "~").strip()


BACK_MATTER_LABELS = ("NOTES TO PAGES", "SOURCE NOTES", "BIBLIOGRAPHY", "INDEX", "APPENDIX")
# A numbered list entry: "23. Mueller to ...". Eight or more on a page means endnotes, a
# contents page, or chart labels; prose pages with a short numbered list have fewer.
NUMBERED_ENTRY = re.compile(r"(?:^|\s)\d{1,3}\. [A-Z\"]")
STAMP = re.compile(
    r"ORIGINAL PAGE\s*\d*\s*(?:IS OF POOR QUALITY|BLACK AND WHITE PHOTOGRAPH|COLOR PHOTOGRAPH)?"
)
# A footnote number glued to the end of a word's punctuation: "director.65 " -> "director. "
# (Fixed-width lookbehind: a lowercase letter, then the punctuation mark.)
FOOTNOTE = re.compile(r"(?<=[a-z][.,;:!?'\")])\d{1,3}(?=\s|$)")


def is_back_matter(page: str) -> bool:
    """Notes, bibliography, index, appendix tables: lists of names and numbers, not prose."""
    if any(label in page[:40].upper() for label in BACK_MATTER_LABELS):
        return True
    if len(NUMBERED_ENTRY.findall(page)) >= 8:
        return True
    return sum(ch.isdigit() for ch in page) / max(len(page), 1) > 0.12  # indexes, tables


def remove_scan_stamps(text: str) -> str:
    """Drop the 'ORIGINAL PAGE ...' stamps NASA's scanning added to photo pages."""
    return STAMP.sub("", text)


def remove_footnote_markers(text: str) -> str:
    return FOOTNOTE.sub("", text)


def included_urls() -> set[str]:
    with LICENSE_LOG.open(encoding="utf-8", newline="") as f:
        return {row["url"] for row in csv.DictReader(f) if row["decision"] == "include"}


def download(url: str, path: Path) -> None:
    """Fetch once and cache: these PDFs are ~200 MB each."""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    context = ssl.create_default_context(cafile=certifi.where())
    partial = path.with_suffix(".part")  # so an interrupted download is never mistaken
    with urllib.request.urlopen(url, context=context) as response, partial.open("wb") as f:
        shutil.copyfileobj(response, f)
    partial.rename(path)


def main() -> None:
    allowed = included_urls()
    TEXT_DIR.mkdir(parents=True, exist_ok=True)
    for book_id, ntrs_id in BOOKS:
        url = pdf_url(ntrs_id)
        if url not in allowed:
            print(f"{book_id}: skipped, no 'include' row in {LICENSE_LOG}")
            continue
        pdf_path = RAW_DIR / f"{book_id}.pdf"
        download(url, pdf_path)
        pages = [page.extract_text() or "" for page in PdfReader(pdf_path).pages]
        prose = [clean_page(p) for p in pages if is_prose(p)]
        kept = [
            " ".join(remove_footnote_markers(remove_scan_stamps(p)).split())  # tidy spaces
            for p in prose
            if not is_back_matter(p)
        ]
        text = "\n\n".join(kept) + "\n"
        (TEXT_DIR / f"{book_id}.txt").write_text(text, encoding="utf-8", newline="\n")
        print(f"{book_id}: kept {len(kept)} of {len(pages)} pages, {len(text):,} characters")


if __name__ == "__main__":
    main()
