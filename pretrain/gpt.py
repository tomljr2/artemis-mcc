"""The Artemis I language model. It starts small and grows into a full GPT step by step.

Current version: token embeddings + position embeddings -> multi-head attention -> MLP ->
output layer, with residual connections and layer normalization. Compared with the bigram
model, each position can now look back at its context (attention) and then process what it
found (MLP).
"""

import torch
import torch.nn.functional as F
from torch import nn

from pretrain.attention import MultiHeadAttention
from pretrain.mlp import FeedForward
from pretrain.norm import LayerNorm


class GPT(nn.Module):
    def __init__(self, vocab_size: int, block_size: int, n_embd: int, n_head: int):
        super().__init__()
        self.block_size = block_size
        # What each character is: one learned n_embd-vector per token id.
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        # Where each character is: one learned n_embd-vector per position 0..block_size-1.
        # Attention by itself is order-blind; adding this gives every position a location.
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.ln1 = LayerNorm(n_embd)
        self.attention = MultiHeadAttention(n_embd, n_head)
        self.ln2 = LayerNorm(n_embd)
        self.mlp = FeedForward(n_embd)
        self.ln_f = LayerNorm(n_embd)  # final norm, before the output layer
        # Turns each position's n_embd numbers into one score per vocabulary entry.
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape
        positions = torch.arange(T, device=idx.device)  # 0, 1, ..., T-1
        x = self.token_embedding(idx) + self.position_embedding(positions)  # (B, T, n_embd)
        # Residual connections: each layer *adds* its result to x instead of replacing it.
        # x is a running record that every layer can read from and write to.
        # "Pre-norm": each layer reads a normalized copy of x, but adds to the raw x, so the
        # residual path itself stays untouched.
        x = x + self.attention(self.ln1(x))  # (B, T, n_embd): mix in context
        x = x + self.mlp(self.ln2(x))  # (B, T, n_embd): process what was gathered
        logits = self.lm_head(self.ln_f(x))  # (B, T, vocab_size)
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
