import torch

from pretrain.dataset import get_batch, train_val_split


def test_split_sizes_follow_the_fraction():
    train, val = train_val_split(list(range(100)), val_fraction=0.1)
    assert len(train) == 90
    assert len(val) == 10


def test_split_keeps_order_with_validation_at_the_end():
    ids = list(range(100))
    train, val = train_val_split(ids, val_fraction=0.1)
    assert train == ids[:90]
    assert val == ids[90:]


def test_batch_has_shape_batch_size_by_block_size():
    x, y = get_batch(torch.arange(100), block_size=8, batch_size=4)
    assert x.shape == (4, 8)
    assert y.shape == (4, 8)


def test_targets_are_the_inputs_shifted_by_one():
    # With data 0, 1, 2, ... the next token after any value v is v + 1.
    x, y = get_batch(torch.arange(100), block_size=8, batch_size=4)
    assert torch.equal(y, x + 1)


def test_windows_stay_inside_the_data():
    # Exactly block_size + 1 tokens leaves room for only one window: the whole thing.
    data = torch.arange(9)
    x, y = get_batch(data, block_size=8, batch_size=3)
    assert torch.equal(x, data[:8].repeat(3, 1))
    assert torch.equal(y, data[1:].repeat(3, 1))
