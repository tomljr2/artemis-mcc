import pytest
import torch

from pretrain.muon import Muon, orthogonalize


@pytest.mark.parametrize("shape", [(64, 64), (32, 128), (128, 32)])
def test_orthogonalize_evens_out_every_direction(shape):
    # A random gradient has strong and weak directions (singular values spread out).
    # Muon's update keeps the directions but gives them all roughly the same strength.
    torch.manual_seed(0)
    k = min(shape)
    u, _ = torch.linalg.qr(torch.randn(shape[0], k))  # k random directions in and out...
    v, _ = torch.linalg.qr(torch.randn(shape[1], k))
    g = u @ torch.diag(torch.logspace(0, 1, k)) @ v.T  # ...with strengths from 1 to 10
    s_after = torch.linalg.svdvals(orthogonalize(g))
    assert s_after.min() > 0.5 and s_after.max() < 1.5


def test_orthogonalize_keeps_the_direction_of_each_part():
    # A diagonal gradient: three independent directions, very different strengths.
    out = orthogonalize(torch.diag(torch.tensor([4.0, 2.0, 1.0])))
    assert torch.all(out.diag() > 0.5) and torch.all(out.diag() < 1.5)  # all pushed ~equally
    assert torch.allclose(out - torch.diag(out.diag()), torch.zeros(3, 3), atol=1e-4)


def test_muon_learns_a_linear_map():
    torch.manual_seed(0)
    target = torch.randn(16, 16)
    w = torch.nn.Parameter(torch.zeros(16, 16))
    opt = Muon([w], lr=0.05)
    x = torch.randn(256, 16)
    losses = []
    for _ in range(200):
        loss = ((x @ w - x @ target) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < 0.05 * losses[0]


def test_muon_only_takes_weight_grids():
    with pytest.raises(ValueError, match="2-D"):
        Muon([torch.nn.Parameter(torch.zeros(5))])
