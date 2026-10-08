"""Training loop: repeatedly measure the loss and nudge the parameters to lower it.

Run from the repo root:  python -m pretrain.train --model gpt     (or --model bigram)
(-m runs it as part of the pretrain package, so its `from pretrain...` imports resolve.)
"""

import argparse
import math
import re
import time
from array import array
from pathlib import Path

import torch
from torch import nn

from pretrain.bigram import BigramModel
from pretrain.bpe import BPETokenizer
from pretrain.checkpoint import save_checkpoint
from pretrain.dataset import get_batch, get_mixed_batch, make_splits, natural_weights
from pretrain.gpt import GPT

TEXT_PATH = Path("data/processed/apollo11_tec.txt")
BOOKS_DIR = Path("data/processed/nasa_books")  # made by: python -m data.prepare_nasa_books
VAL_BOOKS = {"sp-350"}  # Apollo Expeditions to the Moon: held out whole, never trained on
WEB_DIR = Path("data/processed/fineweb_edu")  # made by: python -m data.prepare_fineweb_edu
TOKENIZER_PATH = Path("checkpoints/tokenizer_1024.json")  # python -m pretrain.train_tokenizer
CHECKPOINT_PATH = Path("checkpoints/gpt_nasa.pt")  # git-ignored

# Hyperparameters: settings we choose, as opposed to parameters the model learns.
CONFIGS = {
    "bigram": {
        "block_size": 8,  # context length (the bigram model only uses the last token anyway)
        "batch_size": 32,  # windows per step
        "learning_rate": 1e-2,  # how big each nudge is
        "min_learning_rate": 1e-2,  # same as the peak: no schedule for the bigram
        "warmup_steps": 0,
        "max_steps": 3000,
    },
    "gpt": {
        "block_size": 64,  # context length in characters
        "batch_size": 64,
        "learning_rate": 1e-3,  # peak step size; attention is less forgiving than a table
        "min_learning_rate": 1e-4,  # the schedule decays to a tenth of the peak
        "warmup_steps": 1000,  # ramp up over the first 2.5% of training
        "max_grad_norm": 1.0,  # gradient clipping limit (the usual GPT-2 / Llama value)
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
        # still falling: the limit is now training time. 40,000 steps: 1.270 / 1.347.
        # BPE tokens (vocab 1,024), 40,000 steps, per character: 1.184 / 1.279.
        # + warmup/cosine schedule: 1.174 / 1.276 (a tie; train fit improved more).
        # Cleaned books: 1.252 / 1.277. + single-digit numbers in the tokenizer:
        # 1.120 / 1.272 (timestamps are now cut consistently).
        # + FineWeb-Edu web text (--web), 40,000 steps, by NASA share of training windows:
        # natural 1.1% -> 1.365 / 1.302, 20% -> 1.165 / 1.206, 50% -> 1.095 / 1.188,
        # 80% -> 1.087 / 1.204. 50%, 300,000 steps: 1.054 / 1.154 (model-limited now).
    },
}
CONFIGS["gpt-11m"] = {
    # The same recipe, 11x the parameters: with the web text there is now enough data to
    # feed a bigger model. Wider (384 channels) and deeper (6 blocks); 6 heads of 64
    # channels each, as in GPT-2. On the GTX 1080: ~90 ms per step vs ~35 ms.
    **CONFIGS["gpt"],
    "n_embd": 384,
    "n_head": 6,
    "n_layer": 6,
    # --web, 50% NASA, 40,000 steps: best at step 10,000, 1.056 / 1.167 (1M model: 1.095 /
    # 1.188 at 40,000). Then it memorizes the small NASA text: NASA train 0.354, val
    # transcript 1.208, val book 1.321 at step 40,000. Web val keeps improving (1.161).
}
EVAL_INTERVAL = 500  # report losses every this many steps
EVAL_BATCHES = 100  # batches averaged per loss report


@torch.no_grad()
def clip_gradients(model: nn.Module, max_norm: float) -> float:
    """If the gradients' overall size is above max_norm, shrink them all to that size.

    The size (norm) treats every gradient number of every parameter as one long list:
    square root of the sum of squares. Shrinking multiplies every number by the same factor,
    so the direction of the step is unchanged, only its length is capped. Returns the size
    before clipping.
    """
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    norm = torch.sqrt(sum((g**2).sum() for g in grads))
    scale = max_norm / (norm + 1e-6)  # the tiny 1e-6 avoids dividing by zero
    if scale < 1:
        for g in grads:
            g.mul_(scale)
    return norm.item()


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    y: torch.Tensor,
    max_grad_norm: float | None = None,
) -> float:
    """One update: forward pass, backpropagation, parameter nudge. Returns the loss."""
    _, loss = model(x, y)  # 1. forward: how wrong are the predictions?
    optimizer.zero_grad()  # 2. clear gradients left over from the previous step
    loss.backward()  # 3. backprop: d(loss)/d(parameter) for every parameter
    if max_grad_norm is not None:
        clip_gradients(model, max_grad_norm)  # cap the step's size, keep its direction
    optimizer.step()  # 4. move each parameter a little in the loss-lowering direction
    return loss.item()


def learning_rate(
    step: int, max_lr: float, min_lr: float, warmup_steps: int, max_steps: int
) -> float:
    """Step size for this step: a short linear warmup, then a cosine curve down to min_lr.

    Warmup: the model starts random and its first gradients are wild, so begin with small
    steps. Decay: near the end, smaller steps let it settle into a good spot instead of
    bouncing around it.
    """
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / (max_steps - warmup_steps)  # 0 -> 1 over the decay
    return min_lr + (max_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))


def encode_to_tensor(tok: BPETokenizer, text: str) -> torch.Tensor:
    """tok.encode(text), but stored in 2 bytes per token instead of a Python list.

    For hundreds of millions of characters, a list of ids would take several GB: each id is
    a full Python object. A vocabulary of 1,024 fits easily in a 2-byte integer.
    """
    ids = array("h")  # signed 2-byte integers: up to 32,767
    for match in re.finditer(tok.pattern, text):  # one word-like piece at a time
        ids.extend(tok.encode_word(match.group()))
    return torch.frombuffer(ids, dtype=torch.int16).clone()  # clone: own the memory


def loss_per_char(loss_per_token: float, n_chars: int, n_tokens: int) -> float:
    """Convert a per-token loss to per-character, so tokenizers can be compared fairly.

    The loss on a text is its total surprise divided by how many pieces it was cut into.
    Same text, same total surprise, but more characters than tokens: spread it per character.
    """
    return loss_per_token * n_tokens / n_chars


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


def main(
    model_name: str,
    web: bool = False,
    nasa_weight: float | None = None,
    max_steps: int | None = None,
    checkpoint_path: Path = CHECKPOINT_PATH,
) -> None:
    cfg = CONFIGS[model_name]
    if max_steps is not None:
        cfg = {**cfg, "max_steps": max_steps}
    torch.manual_seed(11)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transcript = TEXT_PATH.read_text(encoding="utf-8")
    books = {p.stem: p.read_text(encoding="utf-8") for p in sorted(BOOKS_DIR.glob("*.txt"))}
    train_text, val_texts = make_splits(transcript, books, VAL_BOOKS)
    texts = {"train": train_text, **val_texts}
    if web:  # general English: its own training text, and its own held-out documents
        texts["train web"] = (WEB_DIR / "train.txt").read_text(encoding="utf-8")
        texts["val web"] = (WEB_DIR / "val.txt").read_text(encoding="utf-8")
    tok = BPETokenizer.load(TOKENIZER_PATH)
    splits = {name: encode_to_tensor(tok, text) for name, text in texts.items()}
    n_chars = {name: len(text) for name, text in texts.items()}
    del texts  # hundreds of MB of text no longer needed: the token ids are what we train on
    for name, data in splits.items():
        print(f"{name}: {n_chars[name]:,} characters -> {len(data):,} tokens")
    print("losses below are per character, to compare with the character-level model")

    sources = {"nasa": splits["train"]}
    if web:
        sources["web"] = splits["train web"]
    if nasa_weight is None:
        weights = natural_weights(sources)  # each source in proportion to its size
    else:
        weights = {"nasa": nasa_weight, "web": 1 - nasa_weight}
    print("training mixture: " + ", ".join(f"{n} {w:.1%}" for n, w in weights.items()))

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
            losses = {
                name: loss_per_char(loss, n_chars[name], len(splits[name]))
                for name, loss in losses.items()
            }
            print(
                f"step {step:5d} | "
                + " | ".join(f"{name} {loss:.3f}" for name, loss in losses.items())
                + f" | {time.perf_counter() - start_time:5.0f}s"
            )
            # One number to pick the best model by: the average of the two NASA validation
            # sets. NASA text is the goal; the web text is there to help with it.
            val = (losses["val transcript"] + losses["val book"]) / 2
            # Keep the best model seen so far, not just the last one: if training starts to
            # overfit, the saved copy is still the version that did best on unseen text.
            if model_name != "bigram" and val < best_val_loss:
                best_val_loss = val
                save_checkpoint(checkpoint_path, model, gpt_args, tok, step, best_val_loss)
                print(f"           saved new best to {checkpoint_path}")
        if step < cfg["max_steps"]:
            lr = learning_rate(
                step,
                cfg["learning_rate"],
                cfg["min_learning_rate"],
                cfg["warmup_steps"],
                cfg["max_steps"],
            )
            for group in optimizer.param_groups:  # AdamW reads its step size from here
                group["lr"] = lr
            x, y = get_mixed_batch(sources, weights, block_size, batch_size, device)
            train_step(model, optimizer, x, y, cfg.get("max_grad_norm"))

    # Let it write: start from a newline and sample 500 tokens. eval() switches dropout
    # off; it is only for training.
    model.eval()
    start = torch.tensor([tok.encode("\n")], device=device)
    print("--- generated ---")
    print(tok.decode(model.generate(start, max_new_tokens=500)[0].tolist()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", choices=CONFIGS, default="gpt")
    parser.add_argument("--web", action="store_true", help="also train on FineWeb-Edu text")
    parser.add_argument(
        "--nasa-weight",
        type=float,
        help="share of training windows from NASA text with --web (default: in proportion to size)",
    )
    parser.add_argument("--max-steps", type=int, help="override the config's max_steps")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    args = parser.parse_args()
    main(args.model, args.web, args.nasa_weight, args.max_steps, args.checkpoint)
