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
