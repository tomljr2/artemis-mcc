"""Turning a stream of token ids into training data."""

from collections.abc import Sequence


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
