"""The Artemis I language model: a GPT built from the parts in this package.

token embeddings + position embeddings -> n_layer transformer blocks -> final norm ->
output layer. Each block lets positions gather context (attention) and process it (MLP),
adding the results to a residual stream that runs from the embeddings to the output.
"""

import torch
import torch.nn.functional as F
from torch import nn

from pretrain.block import Block
from pretrain.norm import RMSNorm


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        n_embd: int,
        n_head: int,
        n_layer: int,
        dropout: float = 0.0,  # fraction of values zeroed during training
    ):
        super().__init__()
        self.block_size = block_size
        # What each character is: one learned n_embd-vector per token id.
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        # Where each character is: one learned n_embd-vector per position 0..block_size-1.
        # Attention by itself is order-blind; adding this gives every position a location.
        self.position_embedding = nn.Embedding(block_size, n_embd)
        # The stack: identical in shape, but each block learns its own weights.
        self.blocks = nn.ModuleList(Block(n_embd, n_head, dropout) for _ in range(n_layer))
        self.ln_f = RMSNorm(n_embd)  # final norm, before the output layer
        # Turns each position's n_embd numbers into one score per vocabulary entry.
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape
        positions = torch.arange(T, device=idx.device)  # 0, 1, ..., T-1
        x = self.token_embedding(idx) + self.position_embedding(positions)  # (B, T, n_embd)
        for block in self.blocks:
            x = block(x)  # (B, T, n_embd): each block reads and adds to the residual stream
        logits = self.lm_head(self.ln_f(x))  # (B, T, vocab_size)
        if targets is None:
            return logits, None
        V = logits.shape[-1]
        loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(
        self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0
    ) -> torch.Tensor:
        """Same sampling loop as the bigram model, but the context is cropped to block_size,
        since the position table has no rows beyond that.

        temperature divides the scores before softmax: below 1 it sharpens the odds toward
        the top choice (cautious), above 1 it flattens them (adventurous), 1 leaves them as
        the model learned them."""
        for _ in range(max_new_tokens):
            context = idx[:, -self.block_size :]
            logits, _ = self(context)
            probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)
        return idx
