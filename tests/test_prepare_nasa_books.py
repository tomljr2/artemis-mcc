from data.prepare_nasa_books import (
    clean_page,
    find_running_heads,
    is_back_matter,
    is_prose,
    remove_footnote_markers,
    remove_scan_stamps,
    strip_running_heads,
)

PROSE = (
    "In looking back at the origins and development of the Apollo program, one word\n"
    "that comes to mind is action. From my vantage as Associate Administrator from 1960\n"
    "to 1965, and then as Deputy Administrator from 1965 to 1968, I had an excellent\n"
    "picture of the intricate action processes that comprised the Apollo program. Disparate\n"
    "and numerous, the actions and their companion reactions came together in a remark-\n"
    "ably coordinated and cooperative blending for the goal of placing men on the Moon\n"
    "and bringing them back safely.\n"
) * 2

# What a photo page turns into: scanner noise, a few stray letters.
PHOTO_JUNK = (
    "A\n^^ pai\n5 ►.\nCJI\t , A\t 4 .\t `W\nN UJ\t ^\t ti 4\n\t7c^`L\n^►\t ^^'\n"
    "{ :r\n\t\n4\t\nN \t • --+ p \t i\nN\t !...\n.+'\t tom+ w vJ \t\n"
) * 6


def test_a_page_of_book_text_is_prose():
    assert is_prose(PROSE)


def test_scanner_noise_from_a_photo_page_is_not_prose():
    assert not is_prose(PHOTO_JUNK)


def test_a_short_caption_is_not_prose():
    assert not is_prose("A parked Rover awaits our return\nfrom sleep.")


def test_lines_are_joined_and_split_words_repaired():
    assert clean_page("a remark-\nably coordinated\nand cooperative") == (
        "a remarkably coordinated and cooperative"
    )


def test_pdf_characters_become_plain_ascii():
    # A ligature (one glyph for "fi"), curly quotes, a tab, and doubled spaces.
    assert clean_page("the ﬁrst “go”\tcall  was") == 'the first "go" call was'


def test_notes_bibliography_index_and_appendix_pages_are_back_matter():
    assert is_back_matter("558 NOTES TO PAGES 65-74 1. House Committee on Science, Report...")
    assert is_back_matter("BIBLIOGRAPHY 603 LEY, Willy. Rockets, Missiles, and Space Travel...")
    assert is_back_matter("IndexABMA. See Army Ballistic Missile Agency. ALSEP. See ...")
    assert is_back_matter("ApPENDIXB Launch Complex 39 1. Vehicle Assembly Building ...")


def test_pages_that_are_mostly_numbers_are_back_matter():
    assert is_back_matter("Stages, 44-46 Tests, 49, 52 Transporting, 50, 54 Vibration, 61, 62, 75")


def test_ordinary_prose_with_some_dates_is_kept():
    assert not is_back_matter(PROSE + " In July 1969, Apollo 11 landed on 20 July at 20:17 UTC.")


def test_scan_stamps_are_removed():
    assert remove_scan_stamps("the crew ORIGINAL PAGE BLACK AND WHITE PHOTOGRAPH waited") == (
        "the crew  waited"
    )
    assert remove_scan_stamps("ORIGINAL PAGE IS OF POOR QUALITY The pad") == " The pad"


def test_footnote_numbers_glued_to_punctuation_are_removed():
    assert remove_footnote_markers("to be deputy center director.65 MSC also") == (
        "to be deputy center director. MSC also"
    )


def test_ordinary_numbers_are_not_footnotes():
    text = "Apollo 11. No. 2 tank, p. 4, at 20:17 in 1969."
    assert remove_footnote_markers(text) == text


def test_source_notes_pages_are_back_matter():
    assert is_back_matter("Source Notes CHAPTER 2 1. R. Cargill Hall, Lunar Impact: A History")


def test_a_page_of_numbered_citations_is_back_matter_even_without_a_label():
    # The running header is the book title, so only the shape gives it away.
    # Few digits (under the 12% rule): only the numbered-entries shape gives it away.
    citations = " ".join(
        f'{n}. Mueller to Administrator, "Manned Space Flight Weekly Report on the schedule."'
        for n in range(20, 30)
    )
    assert is_back_matter("Where No Man Has Gone Before " + citations)


BOOK_PAGES = [
    "CHARIOTS FOR APOLLO on Rector and his group revised the plan. 100",
    "MATCHING MODULES AND MISSIONS had met in October for briefings. 101",
    "CHARIOTS FOR APOLLO Shea tried to persuade Wiesner and Golovin. 102",
    "MATCHING MODULES AND MISSIONS additional studies were agreed upon. 103",
    "CHARIOTS FOR APOLLO The crawler received a further boost in June. 104",
    "MATCHING MODULES AND MISSIONS The board met again in December. 105",
    "CHARIOTS FOR APOLLO The lunar module design was frozen that spring. 106",
    "MATCHING MODULES AND MISSIONS The test pilots flew the trainer daily. 107",
    "CHARIOTS FOR APOLLO The schedule slipped by three months that year. 108",
    "MATCHING MODULES AND MISSIONS The contract was signed in November. 109",
]


def test_titles_repeated_at_the_start_of_many_pages_are_running_heads():
    heads = find_running_heads(BOOK_PAGES)
    assert "CHARIOTS FOR APOLLO" in heads["start"]
    assert "MATCHING MODULES AND MISSIONS" in heads["start"]


def test_ordinary_sentence_starts_are_not_running_heads():
    # "The" begins all six pages, but it is an ordinary word, not a title.
    pages = [
        "The crew rested before the burn.",
        "The pad was cleared at dawn.",
        "The booster arrived by barge.",
        "The engineers checked the valves.",
        "The flight plan was revised.",
        "The tracking station lost signal.",
    ]
    assert find_running_heads(pages)["start"] == set()


def test_running_heads_and_page_numbers_are_stripped_from_both_ends():
    heads = {"start": {"MOONPORT", "CHARIOTS FOR APOLLO"}, "end": {"MEN FOR THE MOON", "APOLLO"}}
    assert strip_running_heads("CHARIOTS FOR APOLLO The crawler was slow. 104", heads) == (
        "The crawler was slow."
    )
    assert strip_running_heads("116 MOONPORT A trip by barge or by rail?", heads) == (
        "A trip by barge or by rail?"
    )
    assert strip_running_heads("the requirement had been dropped 146 APOLLO", heads) == (
        "the requirement had been dropped"
    )
    assert strip_running_heads("crashed and burned. MEN FOR THE MOON 147", heads) == (
        "crashed and burned."
    )


def test_a_running_head_does_not_swallow_the_word_that_often_follows_it():
    # The title starts all 12 pages; "The" follows it on 6. "CHARIOTS FOR APOLLO The" is
    # frequent, but it is not a header: headers continue the same way on (nearly) every page.
    pages = [f"CHARIOTS FOR APOLLO The test number {n} went well." for n in range(6)]
    pages += [f"CHARIOTS FOR APOLLO engineers ran test number {n}." for n in range(6)]
    heads = find_running_heads(pages)["start"]
    assert "CHARIOTS FOR APOLLO" in heads
    assert "CHARIOTS FOR APOLLO The" not in heads
