from data.word_pairs import load_pairs, pair_share


def test_word_pairs_come_from_neighbouring_words(tmp_path):
    ref = tmp_path / "ref.txt"
    ref.write_text("current technology. Current technology and current plans", encoding="utf-8")
    assert load_pairs(ref, max_chars=1000, min_count=2) == {("current", "technology")}


def test_pair_share_drops_when_words_are_out_of_order():
    # NTRS's text of a 1986 scan, against pypdf's reading of the same page.
    pairs = {("are", "controlled"), ("controlled", "by"), ("by", "an"), ("an", "operator")}
    assert pair_share("are controlled by an operator", pairs) == 1.0
    assert pair_share("by an operator controlled are", pairs) == 2 / 4
    assert pair_share("", pairs) == 0.0
