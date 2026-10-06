"""Training loop: repeatedly measure the loss and nudge the parameters to lower it.

Run from the repo root:  python -m pretrain.train
(-m runs it as part of the pretrain package, so its `from pretrain...` imports resolve.)
"""

from pathlib import Path

import torch
from torch import nn

from pretrain.bigram import BigramModel
from pretrain.char_tokenizer import CharTokenizer
from pretrain.dataset import get_batch, train_val_split

TEXT_PATH = Path("data/processed/apollo11_tec.txt")

# Hyperparameters: settings we choose, as opposed to parameters the model learns.
BLOCK_SIZE = 8  # context length (the bigram model only uses the last token anyway)
BATCH_SIZE = 32  # windows per step
LEARNING_RATE = 1e-2  # how big each nudge is
MAX_STEPS = 3000
EVAL_INTERVAL = 300  # report losses every this many steps
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


def main() -> None:
    torch.manual_seed(11)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    text = TEXT_PATH.read_text(encoding="utf-8")
    tok = CharTokenizer(text)
    train_data, val_data = train_val_split(torch.tensor(tok.encode(text), dtype=torch.long))

    model = BigramModel(tok.vocab_size).to(device)
    # AdamW: gradient descent that also adapts the step size for each parameter.
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    for step in range(MAX_STEPS + 1):
        if step % EVAL_INTERVAL == 0:
            losses = estimate_loss(
                model, train_data, val_data, BLOCK_SIZE, BATCH_SIZE, EVAL_BATCHES, device
            )
            print(
                f"step {step:5d} | train loss {losses['train']:.3f}"
                f" | val loss {losses['val']:.3f}"
            )
        if step < MAX_STEPS:
            x, y = get_batch(train_data, BLOCK_SIZE, BATCH_SIZE, device)
            train_step(model, optimizer, x, y)

    # What did it learn? The five most likely characters after a few starting characters.
    with torch.no_grad():
        for ch in ("R", "\n", "L"):
            logits, _ = model(torch.tensor([tok.encode(ch)], device=device))
            top = torch.topk(torch.softmax(logits[0, -1], dim=-1), 5)
            probs, ids = top.values.tolist(), top.indices.tolist()
            guesses = [(tok.decode([i]), round(p, 2)) for p, i in zip(probs, ids)]
            print(f"after {ch!r}: {guesses}")


if __name__ == "__main__":
    main()
