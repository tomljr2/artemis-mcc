import torch
from torch import nn

from pretrain.norm import LayerNorm


def test_each_position_comes_out_with_mean_0_and_spread_1():
    x = torch.randn(2, 5, 16) * 7 + 3  # deliberately large and off-center
    out = LayerNorm(16)(x)
    assert torch.allclose(out.mean(dim=-1), torch.zeros(2, 5), atol=1e-5)
    assert torch.allclose(out.std(dim=-1, unbiased=False), torch.ones(2, 5), atol=1e-3)


def test_matches_pytorchs_layer_norm():
    x = torch.randn(2, 5, 16)
    assert torch.allclose(LayerNorm(16)(x), nn.LayerNorm(16)(x), atol=1e-6)
