from data.clean_ntrs import drop_junk_lines, drop_repeated_lines, is_junk_line


def test_scanner_crumbs_and_stray_symbols_are_junk():
    # Real lines from the 1985-1986 scans and a 2004 report.
    for line in ("-;:;,.1_", "<;;'M!....", '_,_"_\'_"_:"_\'_::', "•", "--", "(4)", ""):
        assert is_junk_line(line), line


def test_lines_with_a_word_or_numbers_are_kept():
    for line in (
        "3.4.6 Development Plan ...............................................",
        "1.0E+10",
        "200",
        "(Lead)",
        "the spacecraft was charged",
    ):
        assert not is_junk_line(line), line


def test_blank_lines_are_kept():
    assert not is_junk_line("   ")
    assert drop_junk_lines("Orbit\n-;:,\n\nMoon") == "Orbit\n\nMoon"


def test_a_lone_letter_is_junk_but_a_short_word_is_not():
    for line in ("I", "o", "  a  "):
        assert is_junk_line(line), line
    for line in ("at", "I/O", "5"):
        assert not is_junk_line(line), line


def test_a_line_on_every_page_is_dropped_whatever_its_number():
    # Pages of a 2026 handbook: the same two header lines, then the page's own text.
    pages = [
        f"NASA Spacecraft Conjunction Assessment\nBest Practices Handbook – Volume {v}\n{body}"
        for v, body in ((1, "Orbits"), (1, "Covariance"), (2, "Screening"), (2, "Maneuvers"))
    ]
    text = drop_repeated_lines("\n".join(pages), pages=4)
    assert text == "Orbits\nCovariance\nScreening\nManeuvers"


def test_short_or_one_word_repeats_are_kept():
    # Narrow-column scans put one word per line, so "spacecraft" repeats like a header;
    # table cells and short labels repeat too.
    text = "\n".join(["spacecraft", "12.5", "Two words", "Development"] * 5)
    assert drop_repeated_lines(text, pages=4) == text


def test_rare_lines_are_kept_in_a_long_report():
    # Five times in 100 pages is a section title, not a running header.
    text = "\n".join(["Thermal Control Design"] * 5 + ["body"])
    assert drop_repeated_lines(text, pages=100) == text
