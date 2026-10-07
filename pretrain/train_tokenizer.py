"""Train the BPE tokenizer once and save it, so every training run uses the same one.

Trained on the training split only: the held-out book and transcript tail stay unseen.
Run:  python -m pretrain.train_tokenizer      (~2 minutes)
"""

import time
from pathlib import Path

from pretrain.bpe import BPETokenizer
from pretrain.dataset import make_splits
from pretrain.train import BOOKS_DIR, TEXT_PATH, VAL_BOOKS

# 1,024 tokens: 256 bytes + 768 merges. Small on purpose: every token gets a row in the
# model's input table and its output layer, and our model is small.
VOCAB_SIZE = 1024
TOKENIZER_PATH = Path(f"checkpoints/tokenizer_{VOCAB_SIZE}.json")  # git-ignored


def main() -> None:
    transcript = TEXT_PATH.read_text(encoding="utf-8")
    books = {p.stem: p.read_text(encoding="utf-8") for p in sorted(BOOKS_DIR.glob("*.txt"))}
    train_text, _ = make_splits(transcript, books, VAL_BOOKS)
    start = time.perf_counter()
    tok = BPETokenizer.train(train_text, VOCAB_SIZE)
    tok.save(TOKENIZER_PATH)
    n_tokens = len(tok.encode(train_text))
    print(f"vocab {tok.vocab_size:,}, trained in {time.perf_counter() - start:.0f}s")
    print(
        f"training text: {len(train_text):,} chars -> {n_tokens:,} tokens "
        f"({len(train_text) / n_tokens:.2f} chars per token)"
    )
    print(f"saved {TOKENIZER_PATH}")


if __name__ == "__main__":
    main()
