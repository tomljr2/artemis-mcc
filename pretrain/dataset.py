"""Turning a stream of token ids into training data."""

from collections.abc import Sequence

import torch


def train_val_split(
    ids: Sequence[int], val_fraction: float = 0.1
) -> tuple[Sequence[int], Sequence[int]]:
    """Split token ids into a training part and a held-out validation part.

    The split is by position, not random: training gets the beginning and validation gets
    the end. A random split would scatter validation characters between training ones, so
    the model would be graded on text it had effectively already seen around it.
    """
    n_train = int(len(ids) * (1 - val_fraction))
    return ids[:n_train], ids[n_train:]


def get_batch(
    data: torch.Tensor, block_size: int, batch_size: int, device: str = "cpu"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cut batch_size random windows of block_size tokens out of data.

    Returns inputs x and targets y, both shaped (batch_size, block_size). y is x shifted one
    token to the left: at every position, the target is the token that comes next.
    """
    # Random start positions. Each window needs block_size + 1 tokens (the +1 is the final
    # target), so the last valid start is len(data) - block_size - 1.
    starts = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in starts])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in starts])
    # .long(): token ids may be stored compactly (2 bytes each); the model needs int64.
    return x.to(device).long(), y.to(device).long()


def make_splits(
    transcript: str, books: dict[str, str], val_books: set[str]
) -> tuple[str, dict[str, str]]:
    """Training text, plus named validation texts, split by document.

    Whole books are held out, so no part of a validation book is ever trained on. The
    transcript is split 90/10 by position as before, which keeps its validation set
    identical to the one behind our earlier numbers.
    """
    transcript_train, transcript_val = train_val_split(transcript)
    train_books = [text for name, text in sorted(books.items()) if name not in val_books]
    train = "\n\n".join([transcript_train, *train_books])
    val = {
        "val transcript": transcript_val,
        "val book": "\n\n".join(books[name] for name in sorted(val_books)),
    }
    return train, val


def natural_weights(sources: dict[str, torch.Tensor]) -> dict[str, float]:
    """Each source's share of all the tokens: what plain concatenation would give."""
    total = sum(len(data) for data in sources.values())
    return {name: len(data) / total for name, data in sources.items()}


def get_mixed_batch(
    sources: dict[str, torch.Tensor],
    weights: dict[str, float],
    block_size: int,
    batch_size: int,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Like get_batch, but each window first picks its source, with these probabilities.

    This is a data mixture: the weights decide how much of training each source gets,
    whatever its size. A small source with a big weight is seen many times over.
    """
    names = list(sources)
    probs = torch.tensor([weights[name] for name in names], dtype=torch.float)
    picks = torch.multinomial(probs, batch_size, replacement=True)  # a source per window
    counts = torch.bincount(picks, minlength=len(names)).tolist()
    parts = [
        get_batch(sources[name], block_size, n, device)
        for name, n in zip(names, counts, strict=True)
        if n > 0
    ]
    return torch.cat([x for x, _ in parts]), torch.cat([y for _, y in parts])
