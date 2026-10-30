from data.compare_extractors import line_break_fates, load_vocab, text_stats
from data.ntrs import ntrs_text


def test_words_split_across_a_line_are_counted():
    text = "permits the dis-\nplay of data, and com-  \n  puters too.\nA well-known fact."
    assert text_stats(text, set())["broken"] == 2  # "well-known" is not split across a line


def test_known_is_the_share_of_words_in_the_reference():
    stats = text_stats("Apollo network rl b", {"apollo", "network"})
    assert stats["words"] == 3  # "b" is too short to count; "rl" counts and is unknown
    assert stats["known"] == 2 / 3


def test_empty_text_has_no_known_share():
    assert text_stats("", {"apollo"}) == {"chars": 0, "words": 0, "broken": 0, "known": 0.0}


def test_vocab_keeps_words_seen_often_enough(tmp_path):
    ref = tmp_path / "ref.txt"
    ref.write_text("Orbit orbit ORBIT moon moon sun", encoding="utf-8")
    assert load_vocab(ref, max_chars=1000, min_count=2) == {"orbit", "moon"}


def test_xhtml_from_ntrs_becomes_plain_text():
    raw = (
        '<html xmlns="http://www.w3.org/1999/xhtml">\n<head>\n<meta name="pdf:PDFVersion" '
        'content="1.7"/>\n</head>\n<body><div class="page"><p>Thrust &amp; drag</p>'
        "<p>Orbit</p></div></body></html>"
    )
    text = ntrs_text(raw)
    assert "Thrust & drag\nOrbit" in text
    assert "<" not in text and "PDFVersion" not in text


def test_plain_ntrs_text_is_left_alone():
    plain = "NASA TECHNICAL MEMORANDUM\n<P is a typo in a scan"
    assert ntrs_text(plain) == plain


def test_line_break_fates_and_suspect_joins():
    pypdf = "the dis-\nplay of high-\nvoltage parts and a well-\nknown sub-\nsystem"
    ntrs = "the display of highvoltage parts and a well-known sub system"
    vocab = {"display", "dis", "play", "high", "voltage", "well", "known", "sub", "system"}
    fates, suspects = line_break_fates(pypdf, ntrs, vocab)
    assert fates == {"joined": 2, "hyphen": 1, "neither": 1}
    assert suspects == ["highvoltage"]  # "display" is a real word, so it isn't suspect


def test_a_word_broken_around_a_blank_line_still_counts():
    # Found in NTRS's text of a 1977 scan: "Satel-", a blank line, then "lite".
    assert text_stats("the Satel-\n\nlite was almost lost", set())["broken"] == 1
