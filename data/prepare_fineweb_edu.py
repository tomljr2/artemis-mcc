"""Download general English text (FineWeb-Edu) and turn it into plain training text.

Our NASA text is only ~4M characters: too little to learn English itself from. FineWeb-Edu
is web pages that a classifier scored as educational, a common choice for small models.
We take one file of its 10-billion-token sample and keep the first TARGET_CHARS characters.

Only used if the dataset has an `include` row in data/license_log.csv.

Run:  python -m data.prepare_fineweb_edu
"""

import re
import unicodedata
from pathlib import Path

import pyarrow.parquet as pq

from data.prepare_nasa_books import PLAIN, download, included_urls

DATASET_URL = "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu"
SHARD_URL = f"{DATASET_URL}/resolve/main/sample/10BT/000_00000.parquet"  # ~2 GB
RAW_PATH = Path("data/raw/fineweb_edu/000_00000.parquet")
TEXT_DIR = Path("data/processed/fineweb_edu")
TARGET_CHARS = 300_000_000  # training characters to keep: ~80x our NASA text
VAL_EVERY = 100  # every 100th document is held out whole for validation


def clean_document(text: str) -> str | None:
    """Plain ASCII like our NASA text, or None if the document needs characters beyond it."""
    text = unicodedata.normalize("NFKC", text)
    text = "".join(PLAIN.get(ch, ch) for ch in text).replace("\t", " ")
    text = re.sub(r" *\n *", "\n", text)  # spaces around line breaks
    text = re.sub(r"\n{3,}", "\n\n", text).strip()  # at most one blank line in a row
    if any(not (" " <= ch <= "~" or ch == "\n") for ch in text):
        return None
    return text


def split_documents(docs: list[str], val_every: int = VAL_EVERY) -> tuple[list[str], list[str]]:
    """(train, val), by whole document: every val_every-th document goes to validation."""
    train = [d for i, d in enumerate(docs) if i % val_every != 0]
    val = [d for i, d in enumerate(docs) if i % val_every == 0]
    return train, val


def load_documents() -> tuple[list[str], int]:
    """(cleaned documents, documents read) from the downloaded file, in their original order."""
    docs, n_seen, n_chars = [], 0, 0
    for batch in pq.ParquetFile(RAW_PATH).iter_batches(batch_size=10_000, columns=["text"]):
        for text in batch.column("text").to_pylist():
            n_seen += 1
            doc = clean_document(text)
            if doc is not None:
                docs.append(doc)
                n_chars += len(doc)
        if n_chars >= TARGET_CHARS / (1 - 1 / VAL_EVERY):  # enough left after holding out val
            break
    return docs, n_seen


def main() -> None:
    if DATASET_URL not in included_urls():
        print(f"skipped: no 'include' row for {DATASET_URL} in the license log")
        return
    download(SHARD_URL, RAW_PATH)
    docs, n_seen = load_documents()
    train, val = split_documents(docs)
    TEXT_DIR.mkdir(parents=True, exist_ok=True)
    for name, part in (("train", train), ("val", val)):
        text = "\n\n".join(part) + "\n"
        (TEXT_DIR / f"{name}.txt").write_text(text, encoding="utf-8", newline="\n")
        print(f"{name}: {len(part):,} documents, {len(text):,} characters")
    print(f"kept {len(docs):,} of {n_seen:,} documents (others needed non-ASCII characters)")


if __name__ == "__main__":
    main()
