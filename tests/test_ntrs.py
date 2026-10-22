from data.ntrs import search_url, summarize

# A cut-down copy of a real record (NTRS 20205009024), so the test needs no network.
RECORD = {
    "id": 20205009024,
    "title": "Automating Test Consistency",
    "authorAffiliations": [{"meta": {"author": {"name": "Justin R Tomlinson"}}}],
    "distributionDate": "2020-11-24T05:00:00.0000000+00:00",
    "stiTypeDetails": "Contractor or Grantee Report",
    "center": {"code": "KSC", "name": "Kennedy Space Center"},
    "distribution": "PUBLIC",
    "copyright": {"belongsToUsGov": False, "determinationType": "GOV_PERMITTED"},
    "downloads": [{"name": "Tomlinson_Justin Final Report Fall 2020.docx"}],
    "keywords": ["Software"],  # one of the many fields summarize leaves out
}


def test_summary_keeps_the_fields_the_harvester_needs():
    assert summarize(RECORD) == {
        "id": 20205009024,
        "title": "Automating Test Consistency",
        "authors": ["Justin R Tomlinson"],
        "date": "2020-11-24",
        "type": "Contractor or Grantee Report",
        "center": "Kennedy Space Center",
        "distribution": "PUBLIC",
        "us_gov_work": False,
        "copyright_determination": "GOV_PERMITTED",
        "files": ["Tomlinson_Justin Final Report Fall 2020.docx"],
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
