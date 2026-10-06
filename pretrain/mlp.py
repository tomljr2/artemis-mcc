"""Feed-forward network (MLP): the "thinking" half of a transformer block.

Attention lets positions communicate: each one gathers information from earlier positions.
The MLP then processes what each position gathered, every position on its own.
"""

import torch
from torch import nn


class FeedForward(nn.Module):
    def __init__(self, n_embd: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            # Expand to 4x the channels: more room to compute. (4x is the convention from
            # the original transformer paper.)
            nn.Linear(n_embd, 4 * n_embd),
            # The nonlinearity: negative numbers become 0, positive ones pass through.
            # Without it the two Linear layers would collapse into a single one, since a
            # linear function of a linear function is still linear.
            nn.ReLU(),
            # Shrink back to n_embd channels so the output fits where the input came from.
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # nn.Linear acts on the last dimension only, so each of the T positions is
        # transformed independently with the same weights.
        return self.net(x)
