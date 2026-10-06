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
