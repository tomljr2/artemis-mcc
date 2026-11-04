from data.clean_ntrs import drop_junk_lines, is_junk_line


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
