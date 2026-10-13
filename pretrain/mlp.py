"""Feed-forward network (MLP): the "thinking" half of a transformer block.

Attention lets positions communicate: each one gathers information from earlier positions.
The MLP then processes what each position gathered, every position on its own.
"""

import torch
import torch.nn.functional as F
from torch import nn


class FeedForward(nn.Module):
    """SwiGLU feed-forward (used by Llama, Mistral and most current models), or ReLU².

    The old version was: expand, ReLU (a fixed rule: "drop negatives"), shrink.
    SwiGLU expands twice instead. `up` carries the information; `gate` decides, number by
    number, how much of it gets through, like a learned volume knob. The model learns what
    to let through instead of relying on a fixed rule.

    kind="relu2" is nanochat's choice: the old two-layer shape, with relu(h)^2 as the rule.
    Squaring keeps the "drop negatives" of ReLU but makes strong signals much stronger than
    weak ones (3 -> 9, 0.5 -> 0.25), and it is smooth at 0 unlike ReLU's corner.
    """

    def __init__(self, n_embd: int, dropout: float = 0.0, kind: str = "swiglu"):
        super().__init__()
        if kind not in ("swiglu", "relu2"):
            raise ValueError(f"unknown MLP kind {kind!r}: use 'swiglu' or 'relu2'")
        self.kind = kind
        if kind == "swiglu":
            # Three layers instead of two, so the hidden size is 2/3 of the old 4 * n_embd
            # to keep the parameter count about the same. No biases, as in Llama.
            hidden = 4 * n_embd * 2 // 3
            self.gate = nn.Linear(n_embd, hidden, bias=False)  # how much to let through
        else:
            hidden = 4 * n_embd  # two layers: the classic 4x expansion, same weight count
        self.up = nn.Linear(n_embd, hidden, bias=False)  # what to let through
        self.down = nn.Linear(hidden, n_embd, bias=False)  # shrink back to n_embd
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind == "relu2":
            return self.dropout(self.down(F.relu(self.up(x)).square()))
        # silu (also called swish) is a smooth ReLU: x * sigmoid(x). Near 0 for negative
        # inputs, about x for positive ones, with no sharp corner at 0.
        # nn.Linear acts on the last dimension only, so each of the T positions is
        # transformed independently with the same weights.
        return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))
