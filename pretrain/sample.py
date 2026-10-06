"""Sample text from a saved model, starting from your own prompt.

Run (from the repo root, after `python -m pretrain.train` has saved a checkpoint):
    python -m pretrain.sample --prompt "Eagle, Houston."

No training happens here: we load the weights and run the forward pass in a loop.
"""

import argparse
from pathlib import Path

import torch

from pretrain.char_tokenizer import CharTokenizer
from pretrain.checkpoint import load_checkpoint
from pretrain.gpt import GPT
from pretrain.train import CHECKPOINT_PATH


def continue_text(
    model: GPT, tok: CharTokenizer, prompt: str, max_new_tokens: int, device: str = "cpu"
) -> str:
    """Encode the prompt, let the model write max_new_tokens more, decode the whole thing."""
    idx = torch.tensor([tok.encode(prompt)], device=device)
    return tok.decode(model.generate(idx, max_new_tokens)[0].tolist())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default="\n", help="text the model continues")
    parser.add_argument("--chars", type=int, default=500, help="how many characters to add")
    parser.add_argument("--seed", type=int, default=None, help="fix the dice for repeatable output")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, vocab, info = load_checkpoint(args.checkpoint, device)
    # The saved vocab is every character in id order, so rebuilding a tokenizer from it gives
    # back exactly the same character-to-id mapping used in training.
    tok = CharTokenizer(vocab)
    print(f"loaded {args.checkpoint} (step {info['step']}, val loss {info['val_loss']:.3f})")
    print(continue_text(model, tok, args.prompt, args.chars, device))


if __name__ == "__main__":
    main()
