"""Byte Pair Encoding (BPE): learn a vocabulary of chunks instead of single characters.

Start from the text's raw bytes (UTF-8), so every possible character is covered: ids 0-255.
Then repeat one rule: find the pair of neighbouring tokens that occurs most often and glue
every occurrence into one new token (256, 257, ...). Later merges build on earlier ones,
so common pieces grow into whole words: "o"+"u" -> "ou", "H"+"ou" -> "Hou", ...
"""

import json
import re
from collections import Counter
from itertools import pairwise
from pathlib import Path

# The pre-tokenizer: cut text into word-like pieces before any merging, so merges stay
# inside a piece. Like GPT-2's rule: contractions ('s, 're, ...), letters with an optional
# leading space, digits, punctuation, then whitespace. (GPT-2 writes letters as \p{L}, which
# needs the third-party `regex` module; [^\W\d_] means the same with the built-in `re`.)
# Unlike GPT-2, each digit is its own piece (as in Llama): every number is then cut the same
# way, instead of some two-digit numbers being one token and others two.
WORD_PATTERN = re.compile(r"'(?:s|t|re|ve|m|ll|d)| ?[^\W\d_]+| ?\d| ?(?:[^\s\w]|_)+|\s+(?!\S)|\s+")


def split_words(text: str, pattern: str = WORD_PATTERN.pattern) -> list[str]:
    """'Houston, we' -> ['Houston', ',', ' we']. Joining the pieces gives back the text."""
    return re.findall(pattern, text)  # re keeps compiled patterns cached


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


def train_bpe(
    text: str, num_merges: int, pattern: str = WORD_PATTERN.pattern
) -> dict[tuple[int, int], int]:
    """Learn num_merges merges from text. Returns {pair: new token id}, in the order learned."""
    # Each distinct word once, with how often it occurs: " Houston" appears thousands of
    # times but is processed once per merge. That is what makes this fast.
    words = {tuple(w.encode("utf-8")): n for w, n in Counter(split_words(text, pattern)).items()}
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


class BPETokenizer:
    """Text <-> token ids, using merges learned by train_bpe.

    A tokenizer is two things: the splitting rule (pattern) and the merges. The merges were
    learned on pieces cut by that rule, so the two are always kept, saved and loaded together.
    """

    def __init__(self, merges: dict[tuple[int, int], int], pattern: str = WORD_PATTERN.pattern):
        self.merges = merges
        self.pattern = pattern
        # What each id spells: the 256 single bytes, then each merge = its two halves joined.
        self.vocab = {i: bytes([i]) for i in range(256)}
        for (a, b), new_id in merges.items():
            self.vocab[new_id] = self.vocab[a] + self.vocab[b]
        self.vocab_size = len(self.vocab)
        self._cache: dict[str, list[int]] = {}  # each distinct word is encoded only once

    @classmethod
    def train(
        cls, text: str, vocab_size: int, pattern: str = WORD_PATTERN.pattern
    ) -> "BPETokenizer":
        return cls(train_bpe(text, vocab_size - 256, pattern), pattern)

    def save(self, path: Path) -> None:
        """Save the splitting rule, and the merges as a list in the order learned."""
        path.parent.mkdir(parents=True, exist_ok=True)
        merges = [[a, b, new_id] for (a, b), new_id in self.merges.items()]
        path.write_text(json.dumps({"pattern": self.pattern, "merges": merges}), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "BPETokenizer":
        saved = json.loads(path.read_text(encoding="utf-8"))
        if "pattern" not in saved:
            # Guessing (e.g. today's default) could silently cut text unlike in training.
            raise ValueError(f"{path} has no splitting rule ('pattern'); cannot load it safely")
        return cls({(a, b): new_id for a, b, new_id in saved["merges"]}, saved["pattern"])

    def encode_word(self, word: str) -> list[int]:
        if word not in self._cache:
            ids = list(word.encode("utf-8"))
            # Apply merges in the order they were learned: always the earliest-learned pair
            # present (lowest id), until no pair in the word has a merge.
            while len(ids) >= 2:
                pair = min(pairwise(ids), key=lambda p: self.merges.get(p, float("inf")))
                if pair not in self.merges:
                    break
                ids = merge(ids, pair, self.merges[pair])
            self._cache[word] = ids
        return self._cache[word]

    def encode(self, text: str) -> list[int]:
        return [i for word in split_words(text, self.pattern) for i in self.encode_word(word)]

    def decode(self, ids: list[int]) -> str:
        # errors="replace": a sequence the model invents may cut a character's bytes in half.
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")
