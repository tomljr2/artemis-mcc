import torch
from torch import nn

from pretrain.norm import LayerNorm, RMSNorm


def test_each_position_comes_out_with_mean_0_and_spread_1():
    x = torch.randn(2, 5, 16) * 7 + 3  # deliberately large and off-center
    out = LayerNorm(16)(x)
    assert torch.allclose(out.mean(dim=-1), torch.zeros(2, 5), atol=1e-5)
    assert torch.allclose(out.std(dim=-1, unbiased=False), torch.ones(2, 5), atol=1e-3)


def test_matches_pytorchs_layer_norm():
    x = torch.randn(2, 5, 16)
    assert torch.allclose(LayerNorm(16)(x), nn.LayerNorm(16)(x), atol=1e-6)


def test_rms_norm_gives_each_position_a_root_mean_square_of_1():
    x = torch.randn(2, 5, 16) * 7 + 3
    out = RMSNorm(16)(x)
    rms = out.pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.ones(2, 5), atol=1e-3)


def test_rms_norm_does_not_recenter():
    # Unlike LayerNorm there is no mean subtraction: an all-positive input stays positive.
    x = torch.rand(2, 5, 16) + 1
    assert (RMSNorm(16)(x) > 0).all()


def test_rms_norm_matches_pytorchs_rms_norm():
    x = torch.randn(2, 5, 16)
    assert torch.allclose(RMSNorm(16)(x), nn.RMSNorm(16, eps=1e-5)(x), atol=1e-6)


def test_rms_norm_has_a_scale_but_no_shift():
    assert [name for name, _ in RMSNorm(16).named_parameters()] == ["weight"]
