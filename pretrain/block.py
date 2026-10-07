"""Transformer block: the unit a GPT stacks over and over.

Each block is "communicate, then compute": attention lets positions gather information from
earlier positions, then the MLP processes it, each position on its own. Both read a
normalized copy of the residual stream and add their result back to it.
"""

import torch
from torch import nn

from pretrain.attention import CausalSelfAttention
from pretrain.mlp import FeedForward
from pretrain.norm import RMSNorm


class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = RMSNorm(n_embd)
        self.attention = CausalSelfAttention(n_embd, n_head, dropout)
        self.ln2 = RMSNorm(n_embd)
        self.mlp = FeedForward(n_embd, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.ln1(x))  # communicate
        x = x + self.mlp(self.ln2(x))  # compute
        return x
