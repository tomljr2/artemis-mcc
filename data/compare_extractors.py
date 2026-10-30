"""Compare two ways of getting text out of the 20 benchmark PDFs.

Contenders:
- NTRS's own .txt, made by NASA for each file it holds (downloaded next to the PDF). For
  some recent files it is XHTML rather than plain text; we strip the markup first.
- pypdf, the extractor we used on the Phase 1 books.

For each report we count: characters, words broken across a line ("dis-\\nplay"), and the
share of words that are real English words. "Real" means seen at least MIN_COUNT times in
the first REF_CHARS characters of our FineWeb-Edu training text. Rare technical words
miss that list too, but they miss it equally for both extractors, so the comparison is fair.

Run (after `python -m data.extract_benchmark`):  python -m data.compare_extractors
"""

import re
import time
from collections import Counter
from pathlib import Path

from data.extract_benchmark import BENCHMARK
from data.ntrs import RAW_DIR, ntrs_text, pypdf_text

REF_TEXT = Path("data/processed/fineweb_edu/train.txt")
REF_CHARS = 50_000_000
MIN_COUNT = 3

WORD = re.compile(r"[A-Za-z]+")
# "dis-" at a line end, "play" on the next line, maybe with blank lines between
BROKEN = re.compile(r"[A-Za-z]-[ \t]*\n\s*[a-z]")


def load_vocab(
    path: Path = REF_TEXT, max_chars: int = REF_CHARS, min_count: int = MIN_COUNT
) -> set[str]:
    """Lowercase words seen at least min_count times in the first max_chars of path."""
    with path.open(encoding="utf-8") as f:
        counts = Counter(w.lower() for w in WORD.findall(f.read(max_chars)))
    return {w for w, n in counts.items() if n >= min_count}


def text_stats(text: str, vocab: set[str]) -> dict:
    """chars, words, broken (words split across lines), known (share of words in vocab)."""
    words = [w.lower() for w in WORD.findall(text) if len(w) >= 2]  # "b", "l": OCR crumbs
    return {
        "chars": len(text),
        "words": len(words),
        "broken": len(BROKEN.findall(text)),
        "known": sum(w in vocab for w in words) / len(words) if words else 0.0,
    }


SPLIT = re.compile(r"([A-Za-z]+)-[ \t]*\n\s*([a-z]+)")  # the two halves of a broken word


def line_break_fates(pypdf: str, ntrs: str, vocab: set[str]) -> tuple[Counter, list[str]]:
    """How NTRS wrote each word pypdf left broken across a line.

    Fates: "joined" (dis-/play -> display), "hyphen" (high-/voltage -> high-voltage),
    "both" (each form appears somewhere), "neither". Also returns the suspect joins: joined
    words that aren't in vocab although both halves are ("highvoltage"), likely joined wrongly.
    """
    fates, suspects = Counter(), []
    for left, right in SPLIT.findall(pypdf):
        joined = re.search(rf"\b{left}{right}\b", ntrs) is not None
        hyphen = re.search(rf"\b{left}-{right}\b", ntrs) is not None
        fate = {(True, False): "joined", (False, True): "hyphen", (True, True): "both"}
        fates[fate.get((joined, hyphen), "neither")] += 1
        whole, halves = (left + right).lower(), (left.lower(), right.lower())
        if joined and not hyphen and whole not in vocab and all(h in vocab for h in halves):
            suspects.append(left + right)
    return fates, suspects


def main() -> None:
    vocab = load_vocab()
    print(f"reference: {len(vocab):,} words\n")
    print(
        f"{'NTRS id':12} {'year':>4}  {'chars ntrs / pypdf':>21}  {'broken':>13}  {'known':>11}  {'pypdf s':>7}"
    )
    totals = {"ntrs": Counter(), "pypdf": Counter()}
    fates, suspects = Counter(), []
    for ntrs_id, year, _ in BENCHMARK:
        folder = RAW_DIR / str(ntrs_id)
        pdf = next(folder.glob("*.pdf"))
        start = time.perf_counter()
        texts = {
            "ntrs": ntrs_text(
                next(folder.glob("*.txt")).read_text(encoding="utf-8", errors="replace")
            ),
            "pypdf": pypdf_text(pdf),
        }
        seconds = time.perf_counter() - start
        stats = {name: text_stats(text, vocab) for name, text in texts.items()}
        report_fates, report_suspects = line_break_fates(texts["pypdf"], texts["ntrs"], vocab)
        fates.update(report_fates)
        suspects += report_suspects
        for name, s in stats.items():
            totals[name].update(
                {
                    "chars": s["chars"],
                    "words": s["words"],
                    "broken": s["broken"],
                    "known_words": round(s["known"] * s["words"]),
                }
            )
        n, p = stats["ntrs"], stats["pypdf"]
        print(
            f"{ntrs_id:<12} {year:>4}  {n['chars']:>10,} / {p['chars']:>9,}  {n['broken']:>5} / {p['broken']:>5}"
            f"  {n['known']:.3f}/{p['known']:.3f}  {seconds:>7.1f}"
        )
    print()
    for name, t in totals.items():
        print(
            f"{name:6} chars {t['chars']:>11,}  broken {t['broken']:>6,}  known {t['known_words'] / t['words']:.3f}"
        )
    print(f"\nwords pypdf broke, as NTRS wrote them: {dict(fates.most_common())}")
    print(f"suspect joins ({len(suspects)}): {', '.join(suspects)}")


if __name__ == "__main__":
    main()
