"""Does a text read in order? Count its neighbouring word pairs that ordinary English uses.

Reference: word pairs seen at least MIN_COUNT times in the first REF_CHARS characters of our
FineWeb-Edu training text. A page read in order keeps hitting them ("controlled by", "by
an"); a page read in scrambled order mostly misses, even when every word is real. Found
and measured in Step 16 (data/README.md, "Text extraction").
"""

import re
from collections import Counter
from itertools import pairwise
from pathlib import Path

REF_TEXT = Path("data/processed/fineweb_edu/train.txt")
REF_CHARS = 50_000_000
MIN_COUNT = 3

WORD = re.compile(r"[A-Za-z]+")


def load_pairs(
    path: Path = REF_TEXT, max_chars: int = REF_CHARS, min_count: int = MIN_COUNT
) -> set[tuple[str, str]]:
    """Lowercase word pairs ("current", "technology") seen at least min_count times."""
    with path.open(encoding="utf-8") as f:
        words = [w.lower() for w in WORD.findall(f.read(max_chars))]
    counts = Counter(pairwise(words))
    return {pair for pair, n in counts.items() if n >= min_count}


def pair_share(text: str, pairs: set[tuple[str, str]]) -> float:
    """Share of neighbouring word pairs found in pairs: low when words are out of order.

    Word counts can't see reading order ("the electromechanical capability or mobility
    provides" has only real words); pairs can.
    """
    words = [w.lower() for w in WORD.findall(text)]
    neighbours = list(pairwise(words))
    return sum(p in pairs for p in neighbours) / len(neighbours) if neighbours else 0.0
