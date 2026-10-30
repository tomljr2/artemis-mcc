"""Read one report's catalogue record from the NASA Technical Reports Server (NTRS).

NTRS (https://ntrs.nasa.gov) holds NASA's technical reports. Each report has a public
JSON record: title, authors, date, and the fields that say who owns it and whether it
may be shared. The harvester will decide from these records which reports to download.

Run:  python -m data.ntrs 19670009662        # one report
      python -m data.ntrs --search apollo     # tally the first 100 matches
      python -m data.ntrs --search apollo --all   # tally every match (~1 request/s)
      python -m data.ntrs 19670009662 --download  # save its PDF and NTRS's own text
"""

import argparse
import csv
import html
import json
import logging
import re
import ssl
import time
import urllib.parse
import urllib.request
from collections import Counter
from collections.abc import Callable, Iterator
from pathlib import Path

import certifi
from pypdf import PdfReader

from data.prepare_nasa_books import download

SITE = "https://ntrs.nasa.gov"
API_URL = f"{SITE}/api/citations"
RAW_DIR = Path("data/raw/ntrs")
LICENSE_LOG = Path("data/license_log.csv")

# What each copyright determination means for us; reasons in data/README.md.
RIGHTS_RULE = {
    "GOV_PUBLIC_USE_PERMITTED": "include",  # a US government work: no US copyright
    "PUBLIC_USE_PERMITTED": "exclude",  # someone else owns it; reuse terms unclear
    "MAY_INCLUDE_COPYRIGHT_MATERIAL": "exclude",
    "GOV_PERMITTED": "exclude",  # government use only
    "NO_PERMISSION": "exclude",
    "OTHER": "exclude",
}


# NASA's own wording for the one value we include (from the NTRS web app).
GOV_WORK_LABEL = "Work of the US Gov. Public Use Permitted."


def rights_decision(determination: str | None) -> str:
    """include or exclude, by RIGHTS_RULE. Anything new or missing is excluded."""
    return RIGHTS_RULE.get(determination, "exclude")


def get_json(url: str) -> dict:
    context = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(url, context=context, timeout=60) as response:
        return json.load(response)


def fetch_record(ntrs_id: int) -> dict:
    """The full JSON record of one report, straight from the API."""
    return get_json(f"{API_URL}/{ntrs_id}")


def search_url(query: str, page_size: int = 100, start: int = 0) -> str:
    """One page of search results: page_size records, starting at match number start."""
    params = {"q": query, "page.size": page_size, "page.from": start}
    return f"{API_URL}/search?{urllib.parse.urlencode(params)}"


def search(query: str, page_size: int = 100, start: int = 0) -> tuple[int, list[dict]]:
    """(total number of matches, the records on this page). Records look like fetch_record's."""
    page = get_json(search_url(query, page_size, start))
    return page["stats"]["total"], page["results"]


def search_all(
    query: str,
    page_size: int = 100,
    pause: float = 1.0,
    search_page: Callable[[str, int, int], tuple[int, list[dict]]] = search,
) -> Iterator[dict]:
    """Every record matching query, one page at a time, pausing between requests."""
    start, total = 0, None
    while total is None or start < total:
        if start:
            time.sleep(pause)  # be polite: one request at a time, with a gap
        total, records = search_page(query, page_size, start)
        if not records:  # the index shrank while we were reading it
            return
        yield from records
        start += len(records)


def author_name(affiliation: dict) -> str | None:
    """Some records list an author entry with no name in it."""
    return ((affiliation.get("meta") or {}).get("author") or {}).get("name")


def summarize(record: dict) -> dict:
    """The few fields we need, out of the ~30 in a record."""
    rights = record.get("copyright") or {}  # some records have it as null
    return {
        "id": record["id"],
        "title": record["title"],
        "authors": [
            name for a in record.get("authorAffiliations") or [] if (name := author_name(a))
        ],
        "date": record.get("distributionDate", "")[:10],  # "2020-11-24T05:00..." -> day only
        "type": record.get("stiTypeDetails"),
        "center": (record.get("center") or {}).get("name"),
        "distribution": record.get("distribution"),  # "PUBLIC" or a restriction
        "us_gov_work": rights.get("belongsToUsGov"),
        "copyright_determination": rights.get("determinationType"),
        "files": [d["name"] for d in record.get("downloads", [])],
        # METADATA_ONLY: a catalogue card with no file of its own.
        "disseminated": record.get("disseminated"),
        # A chapter or paper inside a bigger document (proceedings, report): its text is
        # in the parent's file.
        "part_of": next(
            (
                r["id"]
                for r in record.get("related") or []
                if r.get("type") == "ANALYTIC_PRIMARY" and r.get("id")
            ),
            None,
        ),
    }


def file_links(record: dict) -> list[tuple[str, str]]:
    """(PDF url, text url) for each file. NTRS makes its own text of every file it holds."""
    return [
        (SITE + d["links"]["pdf"], SITE + d["links"]["fulltext"])
        for d in record.get("downloads") or []
    ]


def download_report(record: dict, raw_dir: Path = RAW_DIR) -> list[Path]:
    """Save a report's PDFs and NTRS texts under raw_dir/<id>/, if the rights rule allows."""
    determination = (record.get("copyright") or {}).get("determinationType")
    if rights_decision(determination) != "include":
        raise PermissionError(f"NTRS {record['id']}: {determination} is excluded (data/README.md)")
    saved = []
    for pdf_url, text_url in file_links(record):
        for url in (pdf_url, text_url):
            path = raw_dir / str(record["id"]) / urllib.parse.unquote(url.rsplit("/", 1)[1])
            download(url, path)
            saved.append(path)
    return saved


def ntrs_text(raw: str) -> str:
    """NTRS's text, minus the XHTML wrapping some files come in."""
    if not raw.lstrip().startswith("<html"):
        return raw
    raw = re.sub(r"(?s)<head>.*?</head>", "", raw)  # PDF metadata, not the report
    raw = re.sub(r"</(?:p|div|h\d|li|tr)>", "\n", raw)  # block ends become line breaks
    return html.unescape(re.sub(r"<[^>]+>", "", raw))


def pypdf_text(pdf: Path) -> str:
    logging.getLogger("pypdf").setLevel(logging.ERROR)  # some PDFs warn hundreds of times
    return "\n".join(page.extract_text() or "" for page in PdfReader(pdf).pages)


def report_text(ntrs_id: int, raw_dir: Path = RAW_DIR) -> str:
    """A downloaded report's text: NTRS's own where it has some, else pypdf's.

    NTRS's text won the extractor benchmark (Steps 10-11 in the Flight Log): the same amount
    of text, slightly more real words, 158 words broken across lines against pypdf's 1,106.
    """
    parts = []
    for pdf in sorted((raw_dir / str(ntrs_id)).glob("*.pdf")):
        # NTRS names its text "X.txt" for older files, "X.pdf.txt" for newer ones.
        txts = [
            t for t in (pdf.with_suffix(".txt"), pdf.with_name(pdf.name + ".txt")) if t.exists()
        ]
        text = ntrs_text(txts[0].read_text(encoding="utf-8", errors="replace")) if txts else ""
        parts.append(text if text.strip() else pypdf_text(pdf))
    return "\n\n".join(parts)


def license_row(record: dict, retrieved: str) -> dict:
    """The license-log row for one NTRS report: our decision and NASA's words behind it."""
    determination = (record.get("copyright") or {}).get("determinationType")
    decision = rights_decision(determination)
    publication = (record.get("publications") or [{}])[0]
    year = (publication.get("publicationDate") or "")[:4]
    center = (record.get("center") or {}).get("name")
    publisher = ", ".join(
        part for part in (publication.get("publisher") or "NASA", center, year) if part
    )
    if decision == "include":
        basis = (
            f"NTRS copyright statement: '{GOV_WORK_LABEL}' A US government work (17 U.S.C. 105)."
        )
    else:
        basis = f"NTRS copyright determination {determination}: excluded by the rights rule."
    return {
        "id": f"ntrs-{record['id']}",
        "title": record["title"],
        "url": f"{SITE}/citations/{record['id']}",
        "publisher": f"{publisher} (NTRS {record['id']})",
        "retrieved": retrieved,
        "decision": decision,
        "basis": basis,
        "notes": "Rule in data/README.md. Quoted material and figures may have other owners.",
    }


def append_license_rows(rows: list[dict], log: Path = LICENSE_LOG) -> int:
    """Add rows whose id isn't in the log yet; never change a decision already made."""
    with log.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fields, logged = reader.fieldnames, {row["id"] for row in reader}
    new = [row for row in rows if row["id"] not in logged]
    with log.open("a", encoding="utf-8", newline="") as f:
        csv.DictWriter(f, fields, lineterminator="\n").writerows(new)
    return len(new)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ntrs_id", type=int, nargs="?", help="a report's NTRS id, e.g. 19670009662")
    parser.add_argument(
        "--search", help="search words; tallies the rights fields of the first page"
    )
    parser.add_argument("--all", action="store_true", help="with --search: every page")
    parser.add_argument("--download", action="store_true", help="with an id: save its files")
    args = parser.parse_args()
    if args.search:
        if args.all:
            records = list(search_all(args.search))
            total = len(records)
        else:
            total, records = search(args.search)
        summaries = [summarize(r) for r in records]
        print(f"{total:,} matches for {args.search!r}; tallying {len(summaries):,}")
        for key in ("distribution", "copyright_determination", "us_gov_work", "disseminated"):
            print(f"{key:24} {dict(Counter(s[key] for s in summaries).most_common())}")
        print(f"{'has files':24} {sum(bool(s['files']) for s in summaries)} of {len(summaries)}")
        print(f"{'part of a bigger doc':24} {sum(bool(s['part_of']) for s in summaries)}")
        with_files = Counter(s["copyright_determination"] for s in summaries if s["files"])
        print(f"{'rights, if has files':24} {dict(with_files.most_common())}")
        usable = sum(
            bool(s["files"]) and rights_decision(s["copyright_determination"]) == "include"
            for s in summaries
        )
        print(f"{'usable (files + include)':24} {usable:,}")
    elif args.ntrs_id and args.download:
        for path in download_report(fetch_record(args.ntrs_id)):
            print(f"{path}  {path.stat().st_size:,} bytes")
    elif args.ntrs_id:
        for key, value in summarize(fetch_record(args.ntrs_id)).items():
            print(f"{key:24} {value}")
    else:
        parser.error("give an NTRS id or --search words")


if __name__ == "__main__":
    main()
