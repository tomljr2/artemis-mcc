"""Bigram language model: predicts the next token from the current token alone.

The whole model is one table with a row per token. Row k holds a score (a "logit") for
every possible next token, given that the current token is k. Training will adjust the
table so that likely next tokens score high.
"""

import torch
import torch.nn.functional as F
from torch import nn


class BigramModel(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        # vocab_size rows (current token) x vocab_size columns (score for each next token).
        self.table = nn.Embedding(vocab_size, vocab_size)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # idx is (B, T): B windows of T token ids. Looking each id up in the table gives
        # logits of shape (B, T, vocab_size): next-token scores at every position.
        logits = self.table(idx)
        if targets is None:
            return logits, None

        # cross_entropy expects one row of scores per example, so flatten the B windows of T
        # positions into B*T independent examples.
        B, T, V = logits.shape
        loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """Extend idx (B, T) by max_new_tokens tokens, one at a time.

        Each step: predict scores for the next token, turn them into probabilities, draw one
        token at random according to those probabilities, append it, and repeat.
        """
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            last = logits[:, -1, :]  # only the final position predicts the *next* token
            probs = F.softmax(last, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat([idx, next_token], dim=1)
        return idx
