# data/

NTRS harvester, PDF extraction, cleaning, quality scoring, MinHash dedup, and the per-document
license log. Raw and processed data are git-ignored; code and the license log are tracked.

Phase 2. See [docs/PLAN.md](../docs/PLAN.md#data-plan).

## License log

[`license_log.csv`](license_log.csv) records a decision for every source we consider, used
or not: `include`, `exclude`, or `pending`, with the reason (`basis`). Nothing goes into
training without an `include` row, and `tests/test_license_log.py` checks this for the
text we train on now.
