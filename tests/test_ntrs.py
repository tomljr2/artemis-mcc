import pytest

from data.ntrs import (
    append_license_rows,
    download_report,
    file_links,
    license_row,
    rights_decision,
    search_all,
    search_url,
    summarize,
)

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
    "disseminated": "DOCUMENT_AND_METADATA",
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
        "disseminated": "DOCUMENT_AND_METADATA",
        "part_of": None,
    }


def test_missing_optional_fields_do_not_crash():
    summary = summarize({"id": 1, "title": "Bare record"})
    assert summary["authors"] == [] and summary["files"] == [] and summary["center"] is None


def test_null_rights_and_center_do_not_crash():
    summary = summarize({"id": 1, "title": "Old record", "copyright": None, "center": None})
    assert summary["us_gov_work"] is None and summary["center"] is None


def test_author_entries_without_a_name_are_skipped():
    # Found on a real "apollo" search: an author entry with no name in it.
    record = {"id": 1, "title": "T", "authorAffiliations": [{"meta": {"author": {}}}, {}]}
    assert summarize(record)["authors"] == []


def test_search_url_asks_for_one_page():
    assert search_url("apollo 11", page_size=100, start=200) == (
        "https://ntrs.nasa.gov/api/citations/search?q=apollo+11&page.size=100&page.from=200"
    )


def test_a_chapter_points_to_the_document_that_holds_its_text():
    # NTRS 19670009663, "Apollo network": a catalogue card for one chapter of "Apollo Systems".
    chapter = {
        "id": 19670009663,
        "title": "Apollo network",
        "downloads": [],
        "disseminated": "METADATA_ONLY",
        "related": [
            {"type": "ANALYTIC_PRIMARY", "id": 19670009662, "title": "Apollo Systems"},
            {"type": "ANALYTIC_PRIMARY", "accessionNumber": "N67-18991 08-07"},  # no id
        ],
    }
    summary = summarize(chapter)
    assert summary["files"] == [] and summary["disseminated"] == "METADATA_ONLY"
    assert summary["part_of"] == 19670009662


def test_search_all_walks_every_page_in_order():
    pages = []

    def fake_page(query, page_size, start):  # 5 matches, served 2 at a time
        pages.append(start)
        return 5, [{"id": i} for i in range(start, min(start + page_size, 5))]

    records = list(search_all("apollo", page_size=2, pause=0, search_page=fake_page))
    assert [r["id"] for r in records] == [0, 1, 2, 3, 4]
    assert pages == [0, 2, 4]  # three requests, no extra empty one


def test_search_all_stops_if_the_results_run_out_early():
    # NTRS's count can drop while we read (8,567 one day, 8,566 the next).
    def shrinking(query, page_size, start):
        return 10, [{"id": 0}, {"id": 1}] if start == 0 else []

    assert len(list(search_all("apollo", page_size=2, pause=0, search_page=shrinking))) == 2


def test_only_us_government_works_are_included():
    assert rights_decision("GOV_PUBLIC_USE_PERMITTED") == "include"
    for other in ("PUBLIC_USE_PERMITTED", "MAY_INCLUDE_COPYRIGHT_MATERIAL", "GOV_PERMITTED"):
        assert rights_decision(other) == "exclude"


def test_unknown_or_missing_determinations_are_excluded():
    # If NASA adds a new value, we leave it out until we've read what it means.
    assert rights_decision("SOMETHING_NEW") == "exclude"
    assert rights_decision(None) == "exclude"


def test_each_file_has_a_pdf_and_a_text_link():
    record = {
        "downloads": [
            {
                "links": {
                    "pdf": "/api/citations/19670009662/downloads/19670009662.pdf",
                    "fulltext": "/api/citations/19670009662/downloads/19670009662.txt",
                }
            }
        ]
    }
    assert file_links(record) == [
        (
            "https://ntrs.nasa.gov/api/citations/19670009662/downloads/19670009662.pdf",
            "https://ntrs.nasa.gov/api/citations/19670009662/downloads/19670009662.txt",
        )
    ]


def test_excluded_reports_are_never_downloaded(tmp_path):
    record = {"id": 1, "copyright": {"determinationType": "GOV_PERMITTED"}, "downloads": []}
    with pytest.raises(PermissionError):
        download_report(record, tmp_path)
    assert not any(tmp_path.iterdir())


def test_license_row_records_the_decision_and_nasas_words():
    record = {**RECORD, "publications": [{"publisher": "NASA", "publicationDate": "1966-06-01"}]}
    row = license_row(record, "2026-10-28")
    assert row["id"] == "ntrs-19670009662"
    assert row["url"] == "https://ntrs.nasa.gov/citations/19670009662"
    assert row["publisher"] == "NASA, Goddard Space Flight Center, 1966 (NTRS 19670009662)"
    assert row["decision"] == "include"
    assert "Work of the US Gov. Public Use Permitted." in row["basis"]


def test_license_row_for_an_excluded_report_says_so():
    record = {**RECORD, "copyright": {"determinationType": "GOV_PERMITTED"}}
    row = license_row(record, "2026-10-28")
    assert row["decision"] == "exclude" and "GOV_PERMITTED" in row["basis"]


def test_rows_already_in_the_log_are_left_alone(tmp_path):
    log = tmp_path / "log.csv"
    log.write_text("id,decision\nntrs-1,exclude\n", encoding="utf-8")
    rows = [{"id": "ntrs-1", "decision": "include"}, {"id": "ntrs-2", "decision": "include"}]
    assert append_license_rows(rows, log) == 1
    assert log.read_text(encoding="utf-8") == "id,decision\nntrs-1,exclude\nntrs-2,include\n"
