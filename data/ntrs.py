"""Read one report's catalogue record from the NASA Technical Reports Server (NTRS).

NTRS (https://ntrs.nasa.gov) holds NASA's technical reports. Each report has a public
JSON record: title, authors, date, and the fields that say who owns it and whether it
may be shared. The harvester will decide from these records which reports to download.

Run:  python -m data.ntrs 20205009024
"""

import argparse
import json
import ssl
import urllib.request

import certifi

API_URL = "https://ntrs.nasa.gov/api/citations"


def fetch_record(ntrs_id: int) -> dict:
    """The full JSON record of one report, straight from the API."""
    context = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(f"{API_URL}/{ntrs_id}", context=context, timeout=30) as r:
        return json.load(r)


def summarize(record: dict) -> dict:
    """The few fields we need, out of the ~30 in a record."""
    rights = record.get("copyright", {})
    return {
        "id": record["id"],
        "title": record["title"],
        "authors": [a["meta"]["author"]["name"] for a in record.get("authorAffiliations", [])],
        "date": record.get("distributionDate", "")[:10],  # "2020-11-24T05:00..." -> day only
        "type": record.get("stiTypeDetails"),
        "center": record.get("center", {}).get("name"),
        "distribution": record.get("distribution"),  # "PUBLIC" or a restriction
        "us_gov_work": rights.get("belongsToUsGov"),
        "copyright_determination": rights.get("determinationType"),
        "files": [d["name"] for d in record.get("downloads", [])],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ntrs_id", type=int, help="the report's NTRS id, e.g. 20205009024")
    args = parser.parse_args()
    for key, value in summarize(fetch_record(args.ntrs_id)).items():
        print(f"{key:24} {value}")


if __name__ == "__main__":
    main()
