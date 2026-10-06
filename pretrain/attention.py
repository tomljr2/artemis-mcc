"""Attention, built up one idea at a time.

Step 1 (this file so far): the simplest way for a position to use its context is to
average itself with every earlier position. Written as a matrix multiply, that average is
already attention, just with fixed, equal weights instead of learned ones.
"""

import torch
import torch.nn.functional as F


def causal_average_weights(T: int) -> torch.Tensor:
    """(T, T) matrix whose row t spreads equal weight over positions 0..t.

    Built the same way real attention builds its weights: start from scores (here all
    zero, meaning "no preference"), block out the future with -inf, then softmax. Softmax
    turns -inf into weight 0 and equal scores into equal weights.
    """
    scores = torch.zeros(T, T)
    future = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)  # above the diagonal
    scores = scores.masked_fill(future, float("-inf"))
    return F.softmax(scores, dim=-1)


def causal_average(x: torch.Tensor) -> torch.Tensor:
    """Replace each position of x (B, T, C) with the mean of itself and all earlier ones."""
    T = x.shape[1]
    weights = causal_average_weights(T).to(x.device)
    # (T, T) @ (B, T, C) -> (B, T, C): each output row is a weighted sum of input rows.
    return weights @ x
