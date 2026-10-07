"""Rotary position embeddings (RoPE).

Instead of adding a "where am I" vector to the input once, RoPE turns each query and key by
an angle proportional to its position. Split a vector's channels into pairs; each pair is a
point on a 2-D plane, and position t rotates it by t * frequency. Different pairs use
different frequencies: fast ones tell nearby positions apart, slow ones track long distances.

Why rotation: the dot product of two rotated vectors depends only on the *difference* of
their angles. So a query at position m and a key at position n get a score that depends on
m - n, how far apart they are, not on where they sit. Rotation also keeps lengths unchanged,
so it adds position without changing how strong a vector is.
"""

import torch


def rope_angles(T: int, head_size: int, base: float = 10000.0, device=None):
    """cos and sin of the rotation angles, each (T, head_size // 2).

    Pair i turns at frequency base ** (-2i / head_size): pair 0 by 1 radian per position,
    the last pair very slowly.
    """
    freqs = base ** (-torch.arange(0, head_size, 2, device=device).float() / head_size)
    angles = torch.arange(T, device=device).float()[:, None] * freqs[None, :]  # (T, hs/2)
    return angles.cos(), angles.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate x (..., T, head_size). Channel i is paired with channel i + head_size/2."""
    x1, x2 = x.chunk(2, dim=-1)
    # The 2-D rotation formula, applied to every pair at once.
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
