"""Muon: an optimizer for the model's weight grids (2-D matrices).

A gradient for a weight grid is a mix of directions, a few strong and many weak. Plain
momentum steps follow the strong ones and barely move the weak ones. Muon takes the
momentum, then "orthogonalizes" it: keeps every direction but gives them all about the same
strength, so each step improves the grid in many ways at once.

Embeddings, the output layer and norm weights stay with AdamW (as in nanochat): they are
lookup tables or single vectors, where this reasoning doesn't apply.
This is the original version (https://kellerjordan.github.io/posts/muon/); nanochat adds
refinements on top (Polar Express coefficients, NorMuon, cautious weight decay).
"""

import torch


def orthogonalize(g: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Roughly the nearest matrix whose directions all have strength ~1 (singular values ~1).

    The exact answer needs an SVD, which is slow on a GPU. Instead, a few rounds of a
    polynomial that pushes every singular value toward 1 (Newton-Schulz). These
    coefficients push hard, so values end up between about 0.7 and 1.2, not exactly 1:
    that turns out not to matter.
    """
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g / (g.norm() + 1e-7)  # all singular values now at most 1
    tall = x.size(0) > x.size(1)
    if tall:  # work on the wide orientation: the small x @ x.T is cheaper
        x = x.T
    for _ in range(steps):
        s = x @ x.T
        x = a * x + (b * s + c * s @ s) @ x
    return x.T if tall else x


class Muon(torch.optim.Optimizer):
    """SGD with Nesterov momentum, whose steps are orthogonalized. For 2-D weights only."""

    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95):
        params = list(params)
        if any(p.dim() != 2 for p in params):
            raise ValueError("Muon only takes 2-D weight grids; give the rest to AdamW")
        super().__init__(params, {"lr": lr, "momentum": momentum})

    @torch.no_grad()
    def step(self) -> None:
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                buf = self.state[p].setdefault("momentum", torch.zeros_like(p))
                buf.lerp_(p.grad, 1 - group["momentum"])  # running average of gradients
                g = p.grad.lerp(buf, group["momentum"])  # Nesterov: look one step ahead
                # Orthogonal updates have the same overall size whatever the shape; this
                # scale gives tall grids (more outputs than inputs) proportionally more.
                scale = max(1, p.size(0) / p.size(1)) ** 0.5
                p.add_(orthogonalize(g), alpha=-group["lr"] * scale)
