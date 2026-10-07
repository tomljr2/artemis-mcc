"""Compare our BPE tokenizer (trained on NASA text) with GPT-2's.

Ours is trained on the training split only, so the held-out text stays unseen. Takes ~5 min.
Run:  python -m pretrain.compare_tokenizers
"""

import time

import tiktoken

from pretrain.bpe import BPETokenizer
from pretrain.dataset import make_splits
from pretrain.train import BOOKS_DIR, TEXT_PATH, VAL_BOOKS

transcript = TEXT_PATH.read_text(encoding="utf-8")
books = {p.stem: p.read_text(encoding="utf-8") for p in sorted(BOOKS_DIR.glob("*.txt"))}
train_text, val_texts = make_splits(transcript, books, VAL_BOOKS)

start = time.perf_counter()
ours = BPETokenizer.train(train_text, vocab_size=2048)
seconds = time.perf_counter() - start
print(f"trained ours (vocab 2,048) on {len(train_text):,} chars in {seconds:.0f}s")

gpt2 = tiktoken.get_encoding("gpt2")
print(f"GPT-2 vocab: {gpt2.n_vocab:,}")

print("\n== characters per token on held-out text (higher = fewer tokens) ==")
for name, text in val_texts.items():
    n_ours, n_gpt2 = len(ours.encode(text)), len(gpt2.encode(text))
    print(f"{name:15s} ours {len(text) / n_ours:.2f} | GPT-2 {len(text) / n_gpt2:.2f}")

print("\n== aerospace terms: how each tokenizer cuts them ==")
terms = [
    "Houston",
    "Tranquility",
    "CAPCOM",
    "Columbia",
    "Saturn V",
    "S-IVB",
    "LOX/LH2",
    "Isp",
    "TLI",
    "trans-lunar injection",
    "hypergolic",
    "Grumman",
    "lunar module",
    "N2O4",
    "N₂O₄",
    "delta-V",
    "ΔV",
]
for term in terms:
    t = " " + term  # as it appears mid-sentence
    o = [ours.decode([i]) for i in ours.encode(t)]
    g = [gpt2.decode([i]) for i in gpt2.encode(t)]
    print(f"{term!a:24s} ours {len(o):2d} {o!a}")
    print(f"{'':24s} GPT-2 {len(g):2d} {g!a}")
