"""Clean the text of downloaded NTRS reports, one rule at a time.

Rule 1, junk lines: lines that are scanner crumbs or stray symbols ("-;:;,.1_", "•", "--"),
with no word in them. A line is junk when fewer than half its characters are letters,
digits or spaces, and it has no run of 3+ letters. Lines with a word survive, so contents
lines ("3.4.6 Development Plan ........") and tables of numbers are kept.

Run (after `python -m data.extract_benchmark`):  python -m data.clean_ntrs
"""

import re
from collections import Counter

from data.extract_benchmark import BENCHMARK
from data.ntrs import report_text

WORD3 = re.compile(r"[^\W\d_]{3,}")  # 3+ letters in a row, in any alphabet


def is_junk_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False  # blank lines separate paragraphs; keep them
    clean = sum(c.isalnum() or c == " " for c in s) / len(s)
    return clean < 0.5 and not WORD3.search(s)


def drop_junk_lines(text: str) -> str:
    return "\n".join(line for line in text.splitlines() if not is_junk_line(line))


def main() -> None:
    before, after, lines = Counter(), Counter(), Counter()
    for ntrs_id, year, _ in BENCHMARK:
        text = report_text(ntrs_id)
        decade = year // 10 * 10
        before[decade] += len(text)
        after[decade] += len(drop_junk_lines(text))
        lines[decade] += sum(map(is_junk_line, text.splitlines()))
    print(f"{'decade':7} {'junk lines':>10} {'chars removed':>14} {'of':>10}")
    for decade in sorted(before):
        removed = before[decade] - after[decade]
        print(
            f"{decade}s   {lines[decade]:>10,} {removed:>14,} {before[decade]:>10,}"
            f"  ({removed / before[decade]:.2%})"
        )
    removed = sum(before.values()) - sum(after.values())
    print(
        f"{'all':7} {sum(lines.values()):>10,} {removed:>14,} {sum(before.values()):>10,}"
        f"  ({removed / sum(before.values()):.2%})"
    )


if __name__ == "__main__":
    main()
