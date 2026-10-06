import pytest

from pretrain.char_tokenizer import CharTokenizer


def test_vocabulary_is_the_sorted_unique_characters():
    tok = CharTokenizer("hello")
    assert tok.vocab_size == 4
    # sorted unique characters: e=0, h=1, l=2, o=3
    assert tok.encode("hello") == [1, 0, 2, 2, 3]


def test_decode_reverses_encode():
    text = "11, Houston. Roger. We got a roll program.\n"
    tok = CharTokenizer(text)
    assert tok.decode(tok.encode(text)) == text


def test_unknown_character_is_rejected():
    tok = CharTokenizer("abc")
    with pytest.raises(ValueError, match="not in vocabulary"):
        tok.encode("abz")
