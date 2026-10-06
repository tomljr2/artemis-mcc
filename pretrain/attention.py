"""Attention, built up one idea at a time.

Step 1: the simplest way for a position to use its context is to average itself with
every earlier position. Written as a matrix multiply, that average is already attention,
just with fixed, equal weights instead of learned ones.

Step 2: a self-attention head. Same recipe, but the scores come from comparing what each
position is looking for (its query) with what every earlier position contains (its key).
"""

import math

import torch
import torch.nn.functional as F
from torch import nn


def mask_future(scores: torch.Tensor) -> torch.Tensor:
    """Set every score above the diagonal (a later position) to -inf.

    After softmax, -inf becomes a weight of exactly 0, so no position can use information
    from positions after it.
    """
    T = scores.shape[-1]
    future = torch.triu(torch.ones(T, T, dtype=torch.bool, device=scores.device), diagonal=1)
    return scores.masked_fill(future, float("-inf"))


def causal_average_weights(T: int) -> torch.Tensor:
    """(T, T) matrix whose row t spreads equal weight over positions 0..t.

    Built the same way real attention builds its weights: start from scores (here all
    zero, meaning "no preference"), block out the future, then softmax. Equal scores give
    equal weights.
    """
    return F.softmax(mask_future(torch.zeros(T, T)), dim=-1)


def causal_average(x: torch.Tensor) -> torch.Tensor:
    """Replace each position of x (B, T, C) with the mean of itself and all earlier ones."""
    T = x.shape[1]
    weights = causal_average_weights(T).to(x.device)
    # (T, T) @ (B, T, C) -> (B, T, C): each output row is a weighted sum of input rows.
    return weights @ x


class AttentionHead(nn.Module):
    """One head of causal self-attention."""

    def __init__(self, n_embd: int, head_size: int):
        super().__init__()
        # Three learned projections of each position's n_embd channels:
        self.query = nn.Linear(n_embd, head_size, bias=False)  # what am I looking for?
        self.key = nn.Linear(n_embd, head_size, bias=False)  # what do I contain?
        self.value = nn.Linear(n_embd, head_size, bias=False)  # what do I pass on if picked?
        self.head_size = head_size

    def attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, n_embd) -> (B, T, T): how much each position takes from each earlier one."""
        q = self.query(x)  # (B, T, head_size)
        k = self.key(x)  # (B, T, head_size)
        # Score for (t, s) = dot product of position t's query with position s's key: large
        # when they point the same way. (B, T, hs) @ (B, hs, T) -> (B, T, T).
        scores = q @ k.transpose(-2, -1)
        # Divide by sqrt(head_size) so scores don't grow with head size. Large scores would
        # make softmax put nearly all weight on one position before the model has learned
        # anything, and gradients through such a peaked softmax are tiny.
        scores = scores / math.sqrt(self.head_size)
        return F.softmax(mask_future(scores), dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, n_embd) -> (B, T, head_size): each position's weighted mix of values."""
        return self.attention_weights(x) @ self.value(x)


class MultiHeadAttention(nn.Module):
    """Several attention heads side by side, each free to focus on something different."""

    def __init__(self, n_embd: int, n_head: int):
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError(f"n_embd ({n_embd}) must be divisible by n_head ({n_head})")
        # Split the channels between the heads: 4 heads x 8 channels = 32, the same total
        # size as one 32-wide head.
        head_size = n_embd // n_head
        self.heads = nn.ModuleList(AttentionHead(n_embd, head_size) for _ in range(n_head))
        # Mixes the heads' results together. Without it, each head's findings would stay in
        # its own separate slice of channels.
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Each head returns (B, T, head_size); concatenating along channels gives (B, T, n_embd).
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.proj(out)
