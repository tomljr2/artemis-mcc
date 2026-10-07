import csv
from pathlib import Path

from data.prepare_nasa_books import BOOKS, pdf_url
from pretrain.prepare_apollo11 import URL as APOLLO11_URL

LOG = Path("data/license_log.csv")


def read_log() -> list[dict]:
    with LOG.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_every_entry_has_a_decision_and_a_reason():
    rows = read_log()
    assert rows
    for row in rows:
        assert row["decision"] in {"include", "exclude", "pending"}, row["id"]
        assert row["basis"].strip(), row["id"]


def test_ids_are_unique():
    ids = [row["id"] for row in read_log()]
    assert len(ids) == len(set(ids))


def test_the_text_we_train_on_is_logged_as_included():
    decisions = {row["url"]: row["decision"] for row in read_log()}
    assert decisions.get(APOLLO11_URL) == "include"


def test_every_book_we_prepare_is_logged_as_included():
    decisions = {row["url"]: row["decision"] for row in read_log()}
    for book_id, ntrs_id in BOOKS:
        assert decisions.get(pdf_url(ntrs_id)) == "include", book_id
