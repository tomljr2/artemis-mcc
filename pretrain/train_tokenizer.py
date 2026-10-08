"""Train the BPE tokenizer once and save it, so every training run uses the same one.

Trained on the training split only: the held-out book, transcript tail and web documents
stay unseen.
Run:  python -m pretrain.train_tokenizer                 NASA text only (~2 minutes)
      python -m pretrain.train_tokenizer --web           NASA + web text
"""

import argparse
import time
from pathlib import Path

from pretrain.bpe import BPETokenizer
from pretrain.dataset import make_splits
from pretrain.train import BOOKS_DIR, TEXT_PATH, TOKENIZER_PATH, VAL_BOOKS, WEB_DIR

# 1,024 tokens: 256 bytes + 768 merges. Small on purpose: every token gets a row in the
# model's input table and its output layer, and our model is small.
VOCAB_SIZE = 1024
# With --web: NASA's share of the characters the tokenizer learns from. Matches the share of
# training windows that works best for the 11M model, so the vocabulary fits what it reads.
NASA_SHARE = 0.25
MIXED_TOKENIZER_PATH = Path("checkpoints/tokenizer_1024_mix.json")


def mixed_sample(nasa: str, web: str, nasa_share: float) -> str:
    """All the NASA text, plus enough web text that NASA is nasa_share of the characters.

    Our BPE trainer is plain Python, too slow for all 327M web characters; a sample with
    the right proportions learns nearly the same merges. The web part ends at a document
    break, so no word is cut in half.
    """
    web_chars = int(len(nasa) * (1 - nasa_share) / nasa_share)
    end = web.find("\n\n", web_chars)  # the first document break after the target
    return nasa + "\n\n" + (web if end == -1 else web[:end])


def main(web: bool) -> None:
    transcript = TEXT_PATH.read_text(encoding="utf-8")
    books = {p.stem: p.read_text(encoding="utf-8") for p in sorted(BOOKS_DIR.glob("*.txt"))}
    train_text, _ = make_splits(transcript, books, VAL_BOOKS)
    path = TOKENIZER_PATH
    if web:
        web_text = (WEB_DIR / "train.txt").read_text(encoding="utf-8")
        train_text = mixed_sample(train_text, web_text, NASA_SHARE)
        path = MIXED_TOKENIZER_PATH
    start = time.perf_counter()
    tok = BPETokenizer.train(train_text, VOCAB_SIZE)
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
    parser.add_argument("--web", action="store_true", help="learn from NASA + web text")
    main(parser.parse_args().web)
