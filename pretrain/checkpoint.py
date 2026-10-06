"""Saving and loading trained models.

A checkpoint holds everything needed to rebuild the model without retraining:
  - the learned weights (the model's state_dict: every parameter tensor, by name),
  - the settings that define the model's shape (so we can rebuild an empty one first),
  - the vocabulary (so token ids can be turned back into characters),
  - a little bookkeeping: which step it came from and its validation loss.
"""

from pathlib import Path

import torch

from pretrain.gpt import GPT


def save_checkpoint(
    path: Path, model: GPT, model_args: dict, vocab: str, step: int, val_loss: float
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_args": model_args,
            "state_dict": model.state_dict(),
            "vocab": vocab,
            "step": step,
            "val_loss": val_loss,
        },
        path,
    )


def load_checkpoint(path: Path, device: str = "cpu") -> tuple[GPT, str, dict]:
    """Returns (model in eval mode, vocab, {"step", "val_loss"})."""
    # weights_only=True refuses to run arbitrary code stored in the file: only tensors and
    # plain values (numbers, strings, dicts) are loaded.
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = GPT(**ckpt["model_args"])  # an empty model of the right shape...
    model.load_state_dict(ckpt["state_dict"])  # ...filled with the trained weights
    model.to(device).eval()
    return model, ckpt["vocab"], {"step": ckpt["step"], "val_loss": ckpt["val_loss"]}
