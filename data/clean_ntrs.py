"""Clean the text of downloaded NTRS reports, one rule at a time.

Rule 1, junk lines: lines that are scanner crumbs or stray symbols ("-;:;,.1_", "•", "--"),
with no word in them. A line is junk when fewer than half its characters are letters,
digits or spaces, and it has no run of 3+ letters. Lines with a word survive, so contents
lines ("3.4.6 Development Plan ........") and tables of numbers are kept.

Rule 2, lone letters: a line that is one letter and nothing else. In the scans these are
columns of "I" (printed rules read as letters) and crumbs; in newer reports, letters split
off equations and rotated table headings.

Rule 3, running headers and footers: a line of 2+ words (one of 3+ letters, 10+ characters
in all) that comes back at least once every two pages ("American Institute of Aeronautics
and Astronautics"). Digits don't count, so "Volume 1" and "Volume 2" are the same line.
NTRS's plain text has no page breaks, so repetition is the only sign of a header.

Run (after `python -m data.extract_benchmark`):  python -m data.clean_ntrs
"""

import re
from collections import Counter

from data.extract_benchmark import BENCHMARK
from data.ntrs import page_count, report_text

WORD3 = re.compile(r"[^\W\d_]{3,}")  # 3+ letters in a row, in any alphabet


def is_junk_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False  # blank lines separate paragraphs; keep them
    if len(s) == 1 and s.isalpha():
        return True
    clean = sum(c.isalnum() or c == " " for c in s) / len(s)
    return clean < 0.5 and not WORD3.search(s)


def drop_junk_lines(text: str) -> str:
    return "\n".join(line for line in text.splitlines() if not is_junk_line(line))


def line_shape(line: str) -> str:
    """A line with spacing evened out and every number as "#"."""
    return re.sub(r"\d+", "#", " ".join(line.split()))


def is_header_shape(shape: str) -> bool:
    return len(shape) >= 10 and " " in shape and WORD3.search(shape) is not None


def drop_repeated_lines(text: str, pages: int) -> str:
    """Drop lines that repeat like a running header or footer (rule 3)."""
    lines = text.splitlines()
    counts = Counter(line_shape(line) for line in lines)
    repeated = {s for s, n in counts.items() if n >= max(3, pages / 2) and is_header_shape(s)}
    return "\n".join(line for line in lines if line_shape(line) not in repeated)


def main() -> None:
    chars = {stage: Counter() for stage in ("raw", "rules 1-2", "rule 3")}
    for ntrs_id, year, _ in BENCHMARK:
        decade = year // 10 * 10
        text = report_text(ntrs_id)
        chars["raw"][decade] += len(text)
        text = drop_junk_lines(text)
        chars["rules 1-2"][decade] += len(text)
        chars["rule 3"][decade] += len(drop_repeated_lines(text, page_count(ntrs_id)))
    print(f"{'decade':7} {'chars':>10} {'rules 1-2 remove':>17} {'rule 3 removes':>15}")
    for decade in [*sorted(chars["raw"]), None]:
        raw, junk, header = (
            sum(c.values()) if decade is None else c[decade] for c in chars.values()
        )
        name = "all" if decade is None else f"{decade}s"
        print(
            f"{name:7} {raw:>10,} {raw - junk:>8,} ({(raw - junk) / raw:.2%})"
            f" {junk - header:>6,} ({(junk - header) / raw:.2%})"
        )


if __name__ == "__main__":
    main()
