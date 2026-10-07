"""Saving and loading trained models.

A checkpoint holds everything needed to rebuild the model without retraining:
  - the learned weights (the model's state_dict: every parameter tensor, by name),
  - the settings that define the model's shape (so we can rebuild an empty one first),
  - the tokenizer's merges (so token ids can be turned back into text),
  - a little bookkeeping: which step it came from and its validation loss.
"""

from pathlib import Path

import torch

from pretrain.bpe import BPETokenizer
from pretrain.gpt import GPT


def save_checkpoint(
    path: Path, model: GPT, model_args: dict, tokenizer: BPETokenizer, step: int, val_loss: float
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_args": model_args,
            "state_dict": model.state_dict(),
            # Plain lists of numbers, which the safe loader below accepts.
            "merges": [[a, b, new_id] for (a, b), new_id in tokenizer.merges.items()],
            "step": step,
            "val_loss": val_loss,
        },
        path,
    )


def load_checkpoint(path: Path, device: str = "cpu") -> tuple[GPT, BPETokenizer, dict]:
    """Returns (model in eval mode, tokenizer, {"step", "val_loss"})."""
    # weights_only=True refuses to run arbitrary code stored in the file: only tensors and
    # plain values (numbers, strings, dicts) are loaded.
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = GPT(**ckpt["model_args"])  # an empty model of the right shape...
    model.load_state_dict(ckpt["state_dict"])  # ...filled with the trained weights
    model.to(device).eval()
    tokenizer = BPETokenizer({(a, b): new_id for a, b, new_id in ckpt["merges"]})
    return model, tokenizer, {"step": ckpt["step"], "val_loss": ckpt["val_loss"]}
