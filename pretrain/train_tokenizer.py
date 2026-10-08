"""Train the BPE tokenizer once and save it, so every training run uses the same one.

Trained on the training split only: the held-out book, transcript tail and web documents
stay unseen.
Run:  python -m pretrain.train_tokenizer      4,096 tokens from NASA + web text (~30 minutes)
      python -m pretrain.train_tokenizer --no-web --vocab-size 1024    the first one (~2 min)
"""

import argparse
import time
from pathlib import Path

from pretrain.bpe import BPETokenizer
from pretrain.dataset import make_splits
from pretrain.train import BOOKS_DIR, TEXT_PATH, VAL_BOOKS, WEB_DIR

# 4,096 tokens: 256 bytes + 3,840 merges, enough room for NASA and everyday words both.
# (Every token also gets a row in the model's input table and output layer: 2.4M
# parameters more than 1,024 tokens for gpt-11m.)
VOCAB_SIZE = 4096
# NASA's share of the characters the tokenizer learns from. Matches the share of
# training windows that works best for the 11M model, so the vocabulary fits what it reads.
NASA_SHARE = 0.25


def tokenizer_path(vocab_size: int, web: bool) -> Path:
    """Where a tokenizer is saved: the name says its size and what it learned from."""
    return Path(f"checkpoints/tokenizer_{vocab_size}{'_mix' if web else ''}.json")


def mixed_sample(nasa: str, web: str, nasa_share: float) -> str:
    """All the NASA text, plus enough web text that NASA is nasa_share of the characters.

    Our BPE trainer is plain Python, too slow for all 327M web characters; a sample with
    the right proportions learns nearly the same merges. The web part ends at a document
    break, so no word is cut in half.
    """
    web_chars = int(len(nasa) * (1 - nasa_share) / nasa_share)
    end = web.find("\n\n", web_chars)  # the first document break after the target
    return nasa + "\n\n" + (web if end == -1 else web[:end])


def main(web: bool, vocab_size: int) -> None:
    transcript = TEXT_PATH.read_text(encoding="utf-8")
    books = {p.stem: p.read_text(encoding="utf-8") for p in sorted(BOOKS_DIR.glob("*.txt"))}
    train_text, _ = make_splits(transcript, books, VAL_BOOKS)
    if web:
        web_text = (WEB_DIR / "train.txt").read_text(encoding="utf-8")
        train_text = mixed_sample(train_text, web_text, NASA_SHARE)
    start = time.perf_counter()
    tok = BPETokenizer.train(train_text, vocab_size)
    path = tokenizer_path(vocab_size, web)
    tok.save(path)
    n_tokens = len(tok.encode(train_text))
    print(f"vocab {tok.vocab_size:,}, trained in {time.perf_counter() - start:.0f}s")
    print(
        f"training text: {len(train_text):,} chars -> {n_tokens:,} tokens "
        f"({len(train_text) / n_tokens:.2f} chars per token)"
    )
    print(f"saved {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--web",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="learn from NASA + web text (--no-web: NASA text only)",
    )
    parser.add_argument("--vocab-size", type=int, default=VOCAB_SIZE)
    args = parser.parse_args()
    main(args.web, args.vocab_size)
