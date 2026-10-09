"""Grade any model on the whole web val text the same way, in bits per byte.

Different models split text into different tokens, so loss per token can't be compared.
Bits per byte can: total surprise over the same text, divided by its size. Every val
document is scored from its first token to its last, each exactly once. A model only sees
its own context length, so long documents are read in overlapping windows: each window
starts `stride` tokens after the last, and only its new tokens are scored, so every token
(after the first window) has at least context - stride tokens before it.

Run:  python -m pretrain.score_bpb --checkpoint checkpoints/gpt_11m_nasa25_tok4096_10k.pt
      (nanochat, from its venv, with NANOCHAT_BASE_DIR set as in data/export_nanochat.py)
      python -m pretrain.score_bpb --nanochat ../nanochat-ref --model-tag d6_fineweb
"""

import argparse
import math
import sys
import time
from collections.abc import Callable
from pathlib import Path

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

# The val shard written by data/export_nanochat.py holds our web val documents whole. (A
# plain path, not an import: this also runs in nanochat's venv, which lacks our data tools.)
VAL_SHARD = Path("data/processed/nanochat_fineweb") / "base_data_climbmix" / "shard_00009.parquet"


def document_nats(
    prefix: list[int],
    ids: list[int],
    logits_fn: Callable[[torch.Tensor], torch.Tensor],
    context: int,
    stride: int,
) -> float:
    """Total surprise (nats) of a document's tokens `ids`, given the start signal `prefix`.

    logits_fn maps a 1-D tensor of up to `context` ids to (len, vocab) next-token logits.
    The prefix is the model's own "a document starts here" marker; it isn't scored.
    """
    seq = torch.tensor(prefix + ids)
    nats, scored_until, begin = 0.0, len(prefix), 0  # targets before scored_until are done
    while scored_until < len(seq):
        end = min(begin + context, len(seq) - 1)  # this window's inputs are seq[begin:end]
        logp = F.log_softmax(logits_fn(seq[begin:end]).float(), dim=-1)
        targets = seq[begin + 1 : end + 1]
        new = slice(scored_until - begin - 1, None)  # skip targets an earlier window scored
        nats -= logp[new].gather(1, targets[new, None].to(logp.device)).sum().item()
        scored_until, begin = end + 1, begin + stride
    return nats


def bits_per_byte(nats: float, n_bytes: int) -> float:
    return nats / math.log(2) / n_bytes


def val_documents() -> list[str]:
    return pq.read_table(VAL_SHARD).column("text").to_pylist()


def load_ours(path: Path, device: str):
    """(encode, prefix, logits_fn, context) for one of our checkpoints."""
    from pretrain.checkpoint import load_checkpoint

    model, tok, _ = load_checkpoint(path, device)
    # Our models read documents joined by blank lines, so that is our "new document" signal.
    prefix = tok.encode("\n\n")
    return tok.encode, prefix, lambda x: model(x[None].to(device))[0][0], model.block_size


def load_nanochat(repo: Path, model_tag: str, device: str):
    """(encode, prefix, logits_fn, context) for a nanochat base model."""
    sys.path.insert(0, str(repo.resolve()))
    from nanochat.checkpoint_manager import load_model

    model, tok, meta = load_model("base", torch.device(device), "eval", model_tag=model_tag)
    # nanochat starts every document with its <|bos|> token.
    prefix = [tok.get_bos_token_id()]
    context = meta["model_config"]["sequence_len"]
    return tok.encode, prefix, lambda x: model(x[None].to(device))[0], context


@torch.no_grad()
def main(args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.nanochat:
        encode, prefix, logits_fn, context = load_nanochat(args.nanochat, args.model_tag, device)
        name = f"nanochat {args.model_tag}"
    else:
        encode, prefix, logits_fn, context = load_ours(args.checkpoint, device)
        name = str(args.checkpoint)
    if args.context is not None:  # show the model less than it was trained on
        assert args.context <= context, f"{name} was trained with only {context} tokens"
        context = args.context
    docs = val_documents()[: args.max_docs]
    start = time.perf_counter()
    nats, n_bytes, n_tokens = 0.0, 0, 0
    for doc in docs:
        ids = encode(doc)
        nats += document_nats(prefix, ids, logits_fn, context, context // 2)
        n_bytes += len(doc.encode("utf-8"))
        n_tokens += len(ids)
    print(f"{name}: context {context} tokens, scored in windows overlapping by half")
    print(
        f"{len(docs)} val documents, {n_bytes:,} bytes, {n_tokens:,} tokens "
        f"({n_bytes / n_tokens:.2f} bytes per token)"
    )
    print(f"bits per byte: {bits_per_byte(nats, n_bytes):.4f} ({time.perf_counter() - start:.0f}s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, help="one of our checkpoints")
    parser.add_argument("--nanochat", type=Path, help="path to a nanochat clone")
    parser.add_argument("--model-tag", help="nanochat model tag, e.g. d6_fineweb")
    parser.add_argument("--context", type=int, help="see fewer tokens than the model can")
    parser.add_argument("--max-docs", type=int, help="score only the first N documents")
    main(parser.parse_args())
