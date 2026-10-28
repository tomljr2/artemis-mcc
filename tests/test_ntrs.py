from data.ntrs import search_all, search_url, summarize

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
