import torch

from pretrain.block import Block


def test_block_keeps_the_shape_so_blocks_can_be_stacked():
    assert Block(n_embd=16, n_head=4)(torch.randn(2, 5, 16)).shape == (2, 5, 16)


def test_block_cannot_see_the_future():
    block = Block(n_embd=16, n_head=4)
    x = torch.randn(1, 6, 16)
    changed = x.clone()
    changed[:, 3:] = torch.randn(1, 3, 16)
    with torch.no_grad():
        assert torch.allclose(block(x)[:, :3], block(changed)[:, :3])


def test_a_block_whose_layers_output_zero_is_the_identity():
    # Residual connections: a block that hasn't learned anything leaves x unchanged.
    # (mlp.net[2] is the MLP's second Linear layer.)
    block = Block(n_embd=16, n_head=4)
    with torch.no_grad():
        for layer in (block.attention.proj, block.mlp.net[2]):
            layer.weight.zero_()
            layer.bias.zero_()
        x = torch.randn(2, 5, 16)
        assert torch.allclose(block(x), x)
