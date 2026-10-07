from data.prepare_nasa_books import clean_page, is_prose

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
