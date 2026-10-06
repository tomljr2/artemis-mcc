"""The Artemis I language model. It starts small and grows into a full GPT step by step.

Current version: token embeddings + position embeddings -> one attention head -> output
layer. Compared with the bigram model, each position can now look back at its context.
"""

import torch
import torch.nn.functional as F
from torch import nn

from pretrain.attention import AttentionHead


class GPT(nn.Module):
    def __init__(self, vocab_size: int, block_size: int, n_embd: int):
        super().__init__()
        self.block_size = block_size
        # What each character is: one learned n_embd-vector per token id.
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        # Where each character is: one learned n_embd-vector per position 0..block_size-1.
        # Attention by itself is order-blind; adding this gives every position a location.
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.attention = AttentionHead(n_embd, head_size=n_embd)
        # Turns each position's n_embd numbers into one score per vocabulary entry.
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape
        positions = torch.arange(T, device=idx.device)  # 0, 1, ..., T-1
        x = self.token_embedding(idx) + self.position_embedding(positions)  # (B, T, n_embd)
        x = self.attention(x)  # (B, T, n_embd): each position mixes in its context
        logits = self.lm_head(x)  # (B, T, vocab_size)
        if targets is None:
            return logits, None
        V = logits.shape[-1]
        loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """Same sampling loop as the bigram model, but the context is cropped to block_size,
        since the position table has no rows beyond that."""
        for _ in range(max_new_tokens):
            context = idx[:, -self.block_size :]
            logits, _ = self(context)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)
        return idx
