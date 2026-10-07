from pretrain.bpe import merge, pair_counts, split_words, train_bpe


def test_pair_counts_counts_each_pair_of_neighbours():
    assert pair_counts([1, 2, 1, 2, 3]) == {(1, 2): 2, (2, 1): 1, (2, 3): 1}


def test_merge_replaces_every_occurrence_of_the_pair_with_the_new_token():
    assert merge([1, 2, 1, 2, 3], (1, 2), 256) == [256, 256, 3]


def test_merge_does_not_overlap():
    # "1 1 1" holds the pair (1, 1) twice, but they share the middle 1: only one can merge.
    assert merge([1, 1, 1], (1, 1), 256) == [256, 1]


def test_training_merges_the_most_common_pair_first():
    # Text starts as bytes: "a" = 97, "b" = 98. "ab" is the most common pair.
    assert train_bpe("ab ab ab", num_merges=1) == {(97, 98): 256}


def test_later_merges_build_on_earlier_ones():
    # First "a"+"b" -> 256 ("ab"), then 256+256 -> 257 ("abab").
    assert train_bpe("abababab", num_merges=2) == {(97, 98): 256, (256, 256): 257}


def test_split_words_keeps_letters_together_and_splits_off_punctuation():
    assert split_words("Houston, Tranquility Base here.") == [
        "Houston",
        ",",
        " Tranquility",
        " Base",
        " here",
        ".",
    ]


def test_split_words_keeps_numbers_together_and_splits_off_contractions():
    assert split_words("04 13 we're") == ["04", " 13", " we", "'re"]


def test_split_words_loses_nothing():
    # Joining the pieces must give back the exact text, newlines, underscores and all.
    text = "05 04 16 17 CC\nEagle, Houston.  You're GO_for 1202!\n\n  ok"
    assert "".join(split_words(text)) == text


def test_merges_never_cross_a_word_boundary():
    # Across the whole text, "a" + "." is the most common pair (3 times). But "a" and "."
    # are in different words, so the first merge must be " " + "a" (2 times) instead.
    assert train_bpe("a. a. a.", num_merges=1) == {(32, 97): 256}
