"""Byte Pair Encoding (BPE): learn a vocabulary of chunks instead of single characters.

Start from the text's raw bytes (UTF-8), so every possible character is covered: ids 0-255.
Then repeat one rule: find the pair of neighbouring tokens that occurs most often and glue
every occurrence into one new token (256, 257, ...). Later merges build on earlier ones,
so common pieces grow into whole words: "o"+"u" -> "ou", "H"+"ou" -> "Hou", ...
"""

import re
from collections import Counter
from itertools import pairwise

# The pre-tokenizer: cut text into word-like pieces before any merging, so merges stay
# inside a piece. Like GPT-2's rule: contractions ('s, 're, ...), letters with an optional
# leading space, digits, punctuation, then whitespace. (GPT-2 writes letters as \p{L}, which
# needs the third-party `regex` module; [^\W\d_] means the same with the built-in `re`.)
WORD_PATTERN = re.compile(r"'(?:s|t|re|ve|m|ll|d)| ?[^\W\d_]+| ?\d+| ?(?:[^\s\w]|_)+|\s+(?!\S)|\s+")


def split_words(text: str) -> list[str]:
    """'Houston, we' -> ['Houston', ',', ' we']. Joining the pieces gives back the text."""
    return WORD_PATTERN.findall(text)


def pair_counts(ids: list[int]) -> Counter:
    """How often each pair of neighbours occurs: [1, 2, 1, 2] -> {(1, 2): 2, (2, 1): 1}."""
    return Counter(pairwise(ids))


def merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace every occurrence of pair in ids with new_id, scanning left to right."""
    out = []
    i = 0
    while i < len(ids):
        if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
            out.append(new_id)
            i += 2  # skip both halves of the pair
        else:
            out.append(ids[i])
            i += 1
    return out


def train_bpe(text: str, num_merges: int) -> dict[tuple[int, int], int]:
    """Learn num_merges merges from text. Returns {pair: new token id}, in the order learned."""
    # Each distinct word once, with how often it occurs: " Houston" appears thousands of
    # times but is processed once per merge. That is what makes this fast.
    words = {tuple(w.encode("utf-8")): n for w, n in Counter(split_words(text)).items()}
    merges = {}
    for i in range(num_merges):
        counts = Counter()
        for ids, n in words.items():
            for pair, c in pair_counts(list(ids)).items():
                counts[pair] += c * n  # a pair inside a word counts once per occurrence
        if not counts:
            break  # every word is already a single token
        pair = max(counts, key=counts.get)  # the most common pair (first seen wins a tie)
        new_id = 256 + i
        words = {tuple(merge(list(ids), pair, new_id)): n for ids, n in words.items()}
        merges[pair] = new_id
    return merges
