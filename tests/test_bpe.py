from pretrain.bpe import BPETokenizer, merge, pair_counts, split_words, train_bpe


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


def test_split_words_splits_numbers_into_single_digits_and_splits_off_contractions():
    # Like Llama: every number is cut the same way, whichever numbers were common in
    # training. (Otherwise "03" might be one token while "29" is two.)
    assert split_words("04 13 we're") == ["0", "4", " 1", "3", " we", "'re"]


def test_no_merged_token_ever_contains_two_digits():
    tok = BPETokenizer.train("04 03 29 39 CC\n04 03 31 12 CMP\n" * 50, vocab_size=300)
    pieces = [tok.vocab[i].decode("utf-8") for i in range(256, tok.vocab_size)]
    assert not [p for p in pieces if sum(ch.isdigit() for ch in p) > 1]


def test_split_words_loses_nothing():
    # Joining the pieces must give back the exact text, newlines, underscores and all.
    text = "05 04 16 17 CC\nEagle, Houston.  You're GO_for 1202!\n\n  ok"
    assert "".join(split_words(text)) == text


def test_merges_never_cross_a_word_boundary():
    # Across the whole text, "a" + "." is the most common pair (3 times). But "a" and "."
    # are in different words, so the first merge must be " " + "a" (2 times) instead.
    assert train_bpe("a. a. a.", num_merges=1) == {(32, 97): 256}


def test_tokenizer_encodes_a_learned_pair_as_one_token():
    tok = BPETokenizer.train("ab ab ab", vocab_size=257)
    assert tok.vocab_size == 257
    assert tok.encode(" ab") == [32, 256]


def test_decode_reverses_encode_even_for_characters_never_seen_in_training():
    # Starting from bytes means nothing is ever "unknown": an unseen character is just
    # spelled out as its raw UTF-8 bytes.
    tok = BPETokenizer.train("Houston, Tranquility Base here. " * 20, vocab_size=300)
    text = "Houston: ΔV = 3.1 km/s, café, launch \U0001f680\n\n  Over."
    assert tok.decode(tok.encode(text)) == text


def test_merges_compress_the_text_they_were_trained_on():
    text = "Houston, Tranquility Base here. " * 20
    tok = BPETokenizer.train(text, vocab_size=300)
    assert len(tok.encode(text)) < len(text.encode("utf-8")) / 3


def test_a_saved_tokenizer_loads_back_identical(tmp_path):
    tok = BPETokenizer.train("Houston, Tranquility Base here. " * 20, vocab_size=300)
    path = tmp_path / "tokenizer.json"
    tok.save(path)
    loaded = BPETokenizer.load(path)
    assert loaded.merges == tok.merges
    assert list(loaded.merges) == list(tok.merges)  # same order: merges apply in order
    text = "Tranquility Base, Houston."
    assert loaded.encode(text) == tok.encode(text)
