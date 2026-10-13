import pytest
import torch

from pretrain.mlp import FeedForward


def test_output_keeps_the_input_shape():
    mlp = FeedForward(n_embd=16)
    assert mlp(torch.randn(2, 5, 16)).shape == (2, 5, 16)


def test_each_position_is_processed_on_its_own():
    # Unlike attention, the MLP never looks at other positions: changing positions 1-4
    # must leave position 0's output untouched.
    mlp = FeedForward(n_embd=16)
    x = torch.randn(1, 5, 16)
    changed = x.clone()
    changed[:, 1:] = torch.randn(1, 4, 16)
    with torch.no_grad():
        assert torch.allclose(mlp(x)[:, 0], mlp(changed)[:, 0])


def test_a_closed_gate_lets_nothing_through():
    # SwiGLU: out = down(silu(gate(x)) * up(x)). With the gate's weights at zero the gate
    # reads 0 everywhere, silu(0) = 0, and 0 times anything is 0.
    mlp = FeedForward(n_embd=16)
    with torch.no_grad():
        mlp.gate.weight.zero_()
        assert torch.equal(mlp(torch.randn(2, 5, 16)), torch.zeros(2, 5, 16))


def test_swiglu_costs_about_the_same_as_the_old_relu_mlp():
    # Three layers instead of two, so the hidden size shrinks to 2/3 of 4 * n_embd to keep
    # the parameter count (and so the comparison with the old MLP) fair.
    n = 128
    old_relu_mlp = (n * 4 * n + 4 * n) + (4 * n * n + n)  # two Linear layers with biases
    new = sum(p.numel() for p in FeedForward(n_embd=n).parameters())
    assert abs(new - old_relu_mlp) / old_relu_mlp < 0.01


def test_relu_squared_drops_negatives_and_squares_the_rest():
    # nanochat's MLP: expand, relu(h)^2, shrink. With identity-like layers the hidden
    # values are the inputs themselves, so -2, 0.5, 3 come out as 0, 0.25, 9.
    mlp = FeedForward(n_embd=3, kind="relu2")
    with torch.no_grad():
        mlp.up.weight.copy_(torch.cat([torch.eye(3), torch.zeros(9, 3)]))  # 12 hidden
        mlp.down.weight.copy_(torch.cat([torch.eye(3), torch.zeros(3, 9)], dim=1))
        out = mlp(torch.tensor([[[-2.0, 0.5, 3.0]]]))
    assert torch.allclose(out, torch.tensor([[[0.0, 0.25, 9.0]]]))


def test_relu_squared_costs_the_same_as_swiglu():
    # Two layers of 4 * n_embd vs three of 2/3 of that: the same number of weights, so a
    # comparison between them is about the activation, not the size.
    n = 384
    count = lambda kind: sum(p.numel() for p in FeedForward(n, kind=kind).parameters())
    assert count("relu2") == count("swiglu")


def test_an_unknown_kind_is_refused():
    with pytest.raises(ValueError, match="relu2"):
        FeedForward(n_embd=16, kind="gelu")
