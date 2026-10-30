# data/

NTRS harvester, PDF extraction, cleaning, quality scoring, MinHash dedup, and the per-document
license log. Raw and processed data are git-ignored; code and the license log are tracked.

Phase 2. See [docs/PLAN.md](../docs/PLAN.md#data-plan).

## License log

[`license_log.csv`](license_log.csv) records a decision for every source we consider, used
or not: `include`, `exclude`, or `pending`, with the reason (`basis`). Nothing goes into
training without an `include` row, and `tests/test_license_log.py` checks this for the
text we train on now.

## NTRS rights rule

Each NTRS record carries a copyright determination (`copyright.determinationType`). NASA's
own labels for them come from the NTRS web app; what they mean comes from NASA STI's
[Disclaimers, Copyright Notice, and Terms of Use](https://sti.nasa.gov/disclaimers/) (read
2026-10-28). The harvester follows `RIGHTS_RULE` in [`ntrs.py`](ntrs.py):

| Determination | NASA's label | Rule | Why |
|---|---|---|---|
| `GOV_PUBLIC_USE_PERMITTED` | Work of the US Gov. Public Use Permitted. | include | A US government work: no US copyright (17 U.S.C. 105). Same basis as the Phase 1 books. |
| `PUBLIC_USE_PERMITTED` | Public Use Permitted. | exclude (for now) | Not marked as a government work, so likely someone else owns the copyright. NASA: permission it holds to release such works "does not necessarily transfer to others". Unclear whether it covers training and publishing a dataset. |
| `MAY_INCLUDE_COPYRIGHT_MATERIAL` | Portions of document may include copyright protected material. | exclude | We can't tell which portions automatically. |
| `GOV_PERMITTED` | Use by or on behalf of the US Gov. Permitted. | exclude | Only the government may use it (why SP-4206 was excluded). |
| `NO_PERMISSION` | Content with NO Permission. | exclude | |
| `OTHER`, or none | Other | exclude | No basis. Nearly all are catalogue cards with no file anyway. |

Even an `include` work may quote material owned by others (a photo, a chart, a quotation);
NASA notes that being in a government work doesn't make those public domain. As with the
Phase 1 books, we use the prose.

Harvesting terms ([sti.nasa.gov/harvesting-data-from-ntrs](https://sti.nasa.gov/harvesting-data-from-ntrs/)):
the OpenAPI is NASA's recommended way in, and harvested data should cite **NASA STI
Compliance and Distribution Services** as its source. NASA also offers bulk metadata files,
one per publication year (`ntrs-public-metadata-{year}.ndjson`).

## Text extraction

`report_text` in [`ntrs.py`](ntrs.py) uses the text NTRS makes of each file, and falls back
to [pypdf](https://pypi.org/project/pypdf/) only when NTRS has none. Some NTRS texts are
XHTML; the markup is stripped. Chosen on the 20 reports in
[`extract_benchmark.py`](extract_benchmark.py), measured by
[`compare_extractors.py`](compare_extractors.py) (2026-10-28):

| | Characters | Broken words | Real-word share |
|---|---|---|---|
| NTRS text | 3,983,397 | 158 | 0.959 |
| pypdf | 3,994,411 | 1,106 | 0.957 |

"Broken" means split across a line ("dis-", "play"); "real" means in a 55,561-word list from
our FineWeb-Edu text. NTRS's text scored higher on real words for 16 of the 20 reports. Its
cost: it drops a real hyphen in about 54 places (by hand count), e.g. "highvoltage". Both
carry the scans' OCR errors; cleaning comes next.
