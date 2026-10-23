"""Read one report's catalogue record from the NASA Technical Reports Server (NTRS).

NTRS (https://ntrs.nasa.gov) holds NASA's technical reports. Each report has a public
JSON record: title, authors, date, and the fields that say who owns it and whether it
may be shared. The harvester will decide from these records which reports to download.

Run:  python -m data.ntrs 19670009662        # one report
      python -m data.ntrs --search apollo     # tally the first 100 matches
"""

import argparse
import json
import ssl
import urllib.parse
import urllib.request
from collections import Counter

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


def summarize(record: dict) -> dict:
    """The few fields we need, out of the ~30 in a record."""
    rights = record.get("copyright") or {}  # some records have it as null
    return {
        "id": record["id"],
        "title": record["title"],
        "authors": [a["meta"]["author"]["name"] for a in record.get("authorAffiliations", [])],
        "date": record.get("distributionDate", "")[:10],  # "2020-11-24T05:00..." -> day only
        "type": record.get("stiTypeDetails"),
        "center": (record.get("center") or {}).get("name"),
        "distribution": record.get("distribution"),  # "PUBLIC" or a restriction
        "us_gov_work": rights.get("belongsToUsGov"),
        "copyright_determination": rights.get("determinationType"),
        "files": [d["name"] for d in record.get("downloads", [])],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ntrs_id", type=int, nargs="?", help="a report's NTRS id, e.g. 19670009662")
    parser.add_argument(
        "--search", help="search words; tallies the rights fields of the first page"
    )
    args = parser.parse_args()
    if args.search:
        total, records = search(args.search)
        summaries = [summarize(r) for r in records]
        print(f"{total:,} matches for {args.search!r}; tallying the first {len(summaries)}")
        for key in ("distribution", "copyright_determination", "us_gov_work"):
            print(f"{key:24} {dict(Counter(s[key] for s in summaries).most_common())}")
        print(f"{'has files':24} {sum(bool(s['files']) for s in summaries)} of {len(summaries)}")
    elif args.ntrs_id:
        for key, value in summarize(fetch_record(args.ntrs_id)).items():
            print(f"{key:24} {value}")
    else:
        parser.error("give an NTRS id or --search words")


if __name__ == "__main__":
    main()
