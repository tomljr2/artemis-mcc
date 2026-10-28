"""The 20 NTRS reports we compare PDF text extractors on.

Chosen 2026-10-28 from the first 1,000 matches for "spacecraft": the 293 usable PDFs (files,
and GOV_PUBLIC_USE_PERMITTED), grouped by publication decade, then random.Random(0) picked 3
per decade from the 1960s to the 2010s and 2 from the 2020s. Spread over decades because
the PDFs change with them: typewriter scans in the 1960s, typeset documents later.

The list is written out rather than re-chosen, because NTRS search results change over time.

Run:  python -m data.extract_benchmark    # log a license decision for each, then download
"""

import datetime
import time

from data.ntrs import append_license_rows, download_report, fetch_record, license_row

# (NTRS id, publication year, title)
BENCHMARK = [
    (19660029118, 1966, "A stored program computer for small scientific spacecraft"),
    (19670007177, 1965, "Apollo spacecraft control systems"),
    (19640023654, 1964, "Spacecraft operations manual for epe-d launch"),
    (19780002195, 1977, "Plasma distribution and spacecraft charging modeling near Jupiter"),
    (19780002228, 1977, "System aspects of spacecraft charging"),
    (19780002220, 1977, "Conductive spacecraft materials development program"),
    (19860020413, 1985, "The electrification of spacecraft"),
    (19850014195, 1985, "Laboratory studies of spacecraft response to transient discharge pulses"),
    (19880000702, 1986, "Spacecraft 2000"),
    (19950013364, 1994, "Spacecraft environments interactions: Protecting against the effects"),
    (19940009444, 1993, "Interactions between spacecraft and their environments"),
    (19930013448, 1992, "Spacecraft materials and coatings experiments"),
    (20040111098, 2004, "Representation of the Geosynchronous Plasma Environment for Spacecraft"),
    (20040111031, 2004, "8th Spacecraft Charging Technology Conference"),
    (20030111787, 2003, "Space Environments and Effects (SEE) Program: Spacecraft Charging"),
    (20170007450, 2017, "Spacecraft Alignment Determination and Control for Dual Spacecraft"),
    (20180005514, 2018, "Spacecraft Dormancy Autonomy Analysis for a Crewed Martian Mission"),
    (20140006580, 2013, "Determination of Realistic Fire Scenarios in Spacecraft"),
    (20260000457, 2026, "NASA Spacecraft Conjunction Assessment and Collision Avoidance"),
    (20240007574, 2024, "Spacecraft Priming Analysis and Research Test Apparatus: Number 1"),
]


def main() -> None:
    today = datetime.datetime.now(datetime.UTC).date().isoformat()
    for i, (ntrs_id, _, title) in enumerate(BENCHMARK):
        if i:
            time.sleep(1)  # one request at a time, with a gap
        record = fetch_record(ntrs_id)
        append_license_rows([license_row(record, today)])  # decide first, then download
        size = sum(path.stat().st_size for path in download_report(record))
        print(f"{ntrs_id}  {size / 1e6:6.1f} MB  {title}")


if __name__ == "__main__":
    main()
