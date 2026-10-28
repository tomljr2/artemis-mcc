"""Read one report's catalogue record from the NASA Technical Reports Server (NTRS).

NTRS (https://ntrs.nasa.gov) holds NASA's technical reports. Each report has a public
JSON record: title, authors, date, and the fields that say who owns it and whether it
may be shared. The harvester will decide from these records which reports to download.

Run:  python -m data.ntrs 19670009662        # one report
      python -m data.ntrs --search apollo     # tally the first 100 matches
      python -m data.ntrs --search apollo --all   # tally every match (~1 request/s)
"""

import argparse
import json
import ssl
import time
import urllib.parse
import urllib.request
from collections import Counter
from collections.abc import Callable, Iterator

import certifi

API_URL = "https://ntrs.nasa.gov/api/citations"


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ntrs_id", type=int, nargs="?", help="a report's NTRS id, e.g. 19670009662")
    parser.add_argument(
        "--search", help="search words; tallies the rights fields of the first page"
    )
    parser.add_argument("--all", action="store_true", help="with --search: every page")
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
    elif args.ntrs_id:
        for key, value in summarize(fetch_record(args.ntrs_id)).items():
            print(f"{key:24} {value}")
    else:
        parser.error("give an NTRS id or --search words")


if __name__ == "__main__":
    main()
