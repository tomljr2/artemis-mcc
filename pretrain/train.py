"""Training loop: repeatedly measure the loss and nudge the parameters to lower it.

Run from the repo root:  python -m pretrain.train --model gpt     (or --model bigram)
(-m runs it as part of the pretrain package, so its `from pretrain...` imports resolve.)
"""

import argparse
from pathlib import Path

import torch
from torch import nn

from pretrain.bigram import BigramModel
from pretrain.char_tokenizer import CharTokenizer
from pretrain.dataset import get_batch, train_val_split
from pretrain.gpt import GPT

TEXT_PATH = Path("data/processed/apollo11_tec.txt")

# Hyperparameters: settings we choose, as opposed to parameters the model learns.
CONFIGS = {
    "bigram": {
        "block_size": 8,  # context length (the bigram model only uses the last token anyway)
        "batch_size": 32,  # windows per step
        "learning_rate": 1e-2,  # how big each nudge is
        "max_steps": 3000,
    },
    "gpt": {
        "block_size": 32,  # attention can use context, so give it more
        "batch_size": 32,
        "learning_rate": 1e-3,  # attention is less forgiving of big steps than a table
        "max_steps": 5000,
        "n_embd": 32,  # channels per position
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
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    block_size: int,
    batch_size: int,
    eval_batches: int,
    device: str = "cpu",
) -> dict[str, float]:
    """Average loss over several batches of each split. One batch alone is too noisy."""
    model.eval()
    losses = {}
    for split, data in (("train", train_data), ("val", val_data)):
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

    text = TEXT_PATH.read_text(encoding="utf-8")
    tok = CharTokenizer(text)
    train_data, val_data = train_val_split(torch.tensor(tok.encode(text), dtype=torch.long))

    if model_name == "bigram":
        model = BigramModel(tok.vocab_size)
    else:
        model = GPT(tok.vocab_size, cfg["block_size"], cfg["n_embd"])
    model = model.to(device)
    print(f"{model_name}: {sum(p.numel() for p in model.parameters()):,} parameters")
    # AdamW: gradient descent that also adapts the step size for each parameter.
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"])

    block_size, batch_size = cfg["block_size"], cfg["batch_size"]
    for step in range(cfg["max_steps"] + 1):
        if step % EVAL_INTERVAL == 0:
            losses = estimate_loss(
                model, train_data, val_data, block_size, batch_size, EVAL_BATCHES, device
            )
            print(
                f"step {step:5d} | train loss {losses['train']:.3f}"
                f" | val loss {losses['val']:.3f}"
            )
        if step < cfg["max_steps"]:
            x, y = get_batch(train_data, block_size, batch_size, device)
            train_step(model, optimizer, x, y)

    # Let it write: start from a newline and sample 500 characters.
    start = torch.tensor([tok.encode("\n")], device=device)
    print("--- generated ---")
    print(tok.decode(model.generate(start, max_new_tokens=500)[0].tolist()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", choices=CONFIGS, default="gpt")
    main(parser.parse_args().model)
