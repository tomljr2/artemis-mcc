from data.ntrs import summarize

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
