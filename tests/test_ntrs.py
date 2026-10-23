from data.ntrs import search_url, summarize

# A cut-down copy of a real record (NTRS 19670009662), so the test needs no network.
RECORD = {
    "id": 19670009662,
    "title": "Apollo Systems",
    "authorAffiliations": [{"meta": {"author": {"name": "C Roberts"}}}],
    "distributionDate": "1996-02-12T05:00:00.0000000+00:00",
    "stiTypeDetails": "Technical Memorandum (TM)",
    "center": {"code": "GSFC", "name": "Goddard Space Flight Center"},
    "distribution": "PUBLIC",
    "copyright": {"belongsToUsGov": False, "determinationType": "GOV_PUBLIC_USE_PERMITTED"},
    "downloads": [{"name": "19670009662.pdf"}],
    "keywords": ["Apollo"],  # one of the many fields summarize leaves out
}


def test_summary_keeps_the_fields_the_harvester_needs():
    assert summarize(RECORD) == {
        "id": 19670009662,
        "title": "Apollo Systems",
        "authors": ["C Roberts"],
        "date": "1996-02-12",
        "type": "Technical Memorandum (TM)",
        "center": "Goddard Space Flight Center",
        "distribution": "PUBLIC",
        "us_gov_work": False,
        "copyright_determination": "GOV_PUBLIC_USE_PERMITTED",
        "files": ["19670009662.pdf"],
    }


def test_missing_optional_fields_do_not_crash():
    summary = summarize({"id": 1, "title": "Bare record"})
    assert summary["authors"] == [] and summary["files"] == [] and summary["center"] is None


def test_null_rights_and_center_do_not_crash():
    summary = summarize({"id": 1, "title": "Old record", "copyright": None, "center": None})
    assert summary["us_gov_work"] is None and summary["center"] is None


def test_search_url_asks_for_one_page():
    assert search_url("apollo 11", page_size=100, start=200) == (
        "https://ntrs.nasa.gov/api/citations/search?q=apollo+11&page.size=100&page.from=200"
    )
