"""Training loop: repeatedly measure the loss and nudge the parameters to lower it.

Run from the repo root:  python -m pretrain.train --model gpt     (or --model bigram)
(-m runs it as part of the pretrain package, so its `from pretrain...` imports resolve.)
"""

import argparse
import time
from pathlib import Path

import torch
from torch import nn

from pretrain.bigram import BigramModel
from pretrain.char_tokenizer import CharTokenizer
from pretrain.checkpoint import save_checkpoint
from pretrain.dataset import get_batch, make_splits
from pretrain.gpt import GPT

TEXT_PATH = Path("data/processed/apollo11_tec.txt")
BOOKS_DIR = Path("data/processed/nasa_books")  # made by: python -m data.prepare_nasa_books
VAL_BOOKS = {"sp-350"}  # Apollo Expeditions to the Moon: held out whole, never trained on
CHECKPOINT_PATH = Path("checkpoints/gpt_nasa.pt")  # git-ignored

# Hyperparameters: settings we choose, as opposed to parameters the model learns.
CONFIGS = {
    "bigram": {
        "block_size": 8,  # context length (the bigram model only uses the last token anyway)
        "batch_size": 32,  # windows per step
        "learning_rate": 1e-2,  # how big each nudge is
        "max_steps": 3000,
    },
    "gpt": {
        "block_size": 64,  # context length in characters
        "batch_size": 64,
        "learning_rate": 1e-3,  # attention is less forgiving of big steps than a table
        "max_steps": 40000,  # 10,000 left every loss still falling on transcript + books
        "n_embd": 128,  # channels per position
        "n_head": 4,  # attention heads, each n_embd // n_head = 32 channels wide
        "n_layer": 4,  # transformer blocks stacked
        # Fraction of values zeroed during training, against memorizing. Measured at this
        # size (best val loss): 0.0 -> 1.317 then rising to 1.424 (overfits), 0.1 -> 1.267,
        # 0.2 -> 1.277. On the earlier 57k-param model it only hurt (1.661 -> 1.724).
        "dropout": 0.1,
        # Transcript-only training, best val (seed 11): LayerNorm 1.267, RMSNorm 1.260,
        # + RoPE 1.272, + batched heads 1.270, + SwiGLU 1.268. Evals wobble by ~0.01-0.02,
        # so these are all a tie: the limit was the data, not the model.
        # Transcript + books, 10,000 steps: val transcript 1.333, val book 1.413, all
        # still falling: the limit is now training time.
    },
}
EVAL_INTERVAL = 500  # report losses every this many steps
EVAL_BATCHES = 100  # batches averaged per loss report


def train_step(
    model: nn.Module, optimizer: torch.optim.Optimizer, x: torch.Tensor, y: torch.Tensor
) -> float:
    """One update: forward pass, backpropagation, parameter nudge. Returns the loss."""
    _, loss = model(x, y)  # 1. forward: how wrong are the predictions?
    optimizer.zero_grad()  # 2. clear gradients left over from the previous step
    loss.backward()  # 3. backprop: d(loss)/d(parameter) for every parameter
    optimizer.step()  # 4. move each parameter a little in the loss-lowering direction
    return loss.item()


@torch.no_grad()  # evaluation only: skip the bookkeeping that backprop would need
def estimate_loss(
    model: nn.Module,
    splits: dict[str, torch.Tensor],
    block_size: int,
    batch_size: int,
    eval_batches: int,
    device: str = "cpu",
) -> dict[str, float]:
    """Average loss over several batches of each named split. One batch alone is too noisy."""
    model.eval()
    losses = {}
    for split, data in splits.items():
        total = 0.0
        for _ in range(eval_batches):
            x, y = get_batch(data, block_size, batch_size, device)
            total += model(x, y)[1].item()
        losses[split] = total / eval_batches
    model.train()
    return losses


def main(model_name: str) -> None:
    cfg = CONFIGS[model_name]
    torch.manual_seed(11)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transcript = TEXT_PATH.read_text(encoding="utf-8")
    books = {p.stem: p.read_text(encoding="utf-8") for p in sorted(BOOKS_DIR.glob("*.txt"))}
    train_text, val_texts = make_splits(transcript, books, VAL_BOOKS)
    # The vocabulary must cover every character the model will be asked about, val included.
    tok = CharTokenizer(train_text + "".join(val_texts.values()))
    splits = {
        name: torch.tensor(tok.encode(text), dtype=torch.long)
        for name, text in {"train": train_text, **val_texts}.items()
    }
    for name, data in splits.items():
        print(f"{name}: {len(data):,} characters")
    train_data = splits["train"]

    if model_name == "bigram":
        model = BigramModel(tok.vocab_size)
    else:
        gpt_args = {
            "vocab_size": tok.vocab_size,
            "block_size": cfg["block_size"],
            "n_embd": cfg["n_embd"],
            "n_head": cfg["n_head"],
            "n_layer": cfg["n_layer"],
            "dropout": cfg["dropout"],
        }
        model = GPT(**gpt_args)
    model = model.to(device)
    print(f"{model_name}: {sum(p.numel() for p in model.parameters()):,} parameters")
    # AdamW: gradient descent that also adapts the step size for each parameter.
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"])

    block_size, batch_size = cfg["block_size"], cfg["batch_size"]
    start_time = time.perf_counter()
    best_val_loss = float("inf")
    for step in range(cfg["max_steps"] + 1):
        if step % EVAL_INTERVAL == 0:
            losses = estimate_loss(model, splits, block_size, batch_size, EVAL_BATCHES, device)
            # One number to pick the best model by: the average of the two validation sets.
            losses["val"] = (losses["val transcript"] + losses["val book"]) / 2
            print(
                f"step {step:5d} | train {losses['train']:.3f}"
                f" | val transcript {losses['val transcript']:.3f}"
                f" | val book {losses['val book']:.3f}"
                f" | {time.perf_counter() - start_time:5.0f}s"
            )
            # Keep the best model seen so far, not just the last one: if training starts to
            # overfit, the saved copy is still the version that did best on unseen text.
            if model_name == "gpt" and losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                vocab = tok.decode(list(range(tok.vocab_size)))  # every character, in id order
                save_checkpoint(CHECKPOINT_PATH, model, gpt_args, vocab, step, best_val_loss)
                print(f"           saved new best to {CHECKPOINT_PATH}")
        if step < cfg["max_steps"]:
            x, y = get_batch(train_data, block_size, batch_size, device)
            train_step(model, optimizer, x, y)

    # Let it write: start from a newline and sample 500 characters. eval() switches dropout
    # off; it is only for training.
    model.eval()
    start = torch.tensor([tok.encode("\n")], device=device)
    print("--- generated ---")
    print(tok.decode(model.generate(start, max_new_tokens=500)[0].tolist()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", choices=CONFIGS, default="gpt")
    main(parser.parse_args().model)
