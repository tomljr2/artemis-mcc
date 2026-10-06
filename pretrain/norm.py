"""Normalization: rescale each position's numbers to a standard size before a layer reads them."""

import torch
from torch import nn


class LayerNorm(nn.Module):
    def __init__(self, n_embd: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps  # tiny number that prevents dividing by zero
        # Learned scale and shift, so the model can undo the normalization where it helps.
        # They start as "multiply by 1, add 0", i.e. plain normalization.
        self.weight = nn.Parameter(torch.ones(n_embd))
        self.bias = nn.Parameter(torch.zeros(n_embd))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Statistics over each position's own n_embd channels (the last dimension).
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)  # now average 0, spread 1
        return x_hat * self.weight + self.bias


class RMSNorm(nn.Module):
    """LayerNorm without the mean subtraction and without the shift (used by Llama and most
    current models). It only rescales: divide by the root mean square of the numbers.
    One fewer statistic to compute, and in practice it trains just as well."""

    def __init__(self, n_embd: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(n_embd))  # learned scale, no bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight
