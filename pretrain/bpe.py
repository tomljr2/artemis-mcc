"""Byte Pair Encoding (BPE): learn a vocabulary of chunks instead of single characters.

Start from the text's raw bytes (UTF-8), so every possible character is covered: ids 0-255.
Then repeat one rule: find the pair of neighbouring tokens that occurs most often and glue
every occurrence into one new token (256, 257, ...). Later merges build on earlier ones,
so common pieces grow into whole words: "o"+"u" -> "ou", "H"+"ou" -> "Hou", ...
"""

from collections import Counter
from itertools import pairwise


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
    ids = list(text.encode("utf-8"))
    merges = {}
    for i in range(num_merges):
        counts = pair_counts(ids)
        if not counts:
            break  # the whole text is already one token
        pair = max(counts, key=counts.get)  # the most common pair (first seen wins a tie)
        new_id = 256 + i
        ids = merge(ids, pair, new_id)
        merges[pair] = new_id
    return merges
