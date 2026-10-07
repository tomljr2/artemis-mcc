"""Sample text from a saved model, starting from your own prompt.

Run (from the repo root, after `python -m pretrain.train` has saved a checkpoint):
    python -m pretrain.sample --prompt "Eagle, Houston."

No training happens here: we load the weights and run the forward pass in a loop.
"""

import argparse
from pathlib import Path

import torch

from pretrain.bpe import BPETokenizer
from pretrain.checkpoint import load_checkpoint
from pretrain.gpt import GPT
from pretrain.train import CHECKPOINT_PATH


def continue_text(
    model: GPT,
    tok: BPETokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float = 1.0,
    device: str = "cpu",
) -> str:
    """Encode the prompt, let the model write max_new_tokens more, decode the whole thing."""
    idx = torch.tensor([tok.encode(prompt)], device=device)
    return tok.decode(model.generate(idx, max_new_tokens, temperature)[0].tolist())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default="\n", help="text the model continues")
    parser.add_argument("--tokens", type=int, default=200, help="how many tokens to add")
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="<1 cautious, >1 adventurous"
    )
    parser.add_argument("--seed", type=int, default=None, help="fix the dice for repeatable output")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # The checkpoint carries its own tokenizer, so text is cut exactly as in training.
    model, tok, info = load_checkpoint(args.checkpoint, device)
    print(f"loaded {args.checkpoint} (step {info['step']}, val loss {info['val_loss']:.3f})")
    print(continue_text(model, tok, args.prompt, args.tokens, args.temperature, device))


if __name__ == "__main__":
    main()
