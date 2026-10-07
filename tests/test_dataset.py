import torch

from pretrain.dataset import get_batch, make_splits, train_val_split


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


def test_a_held_out_book_never_reaches_training():
    train, val = make_splits("t" * 100, {"a": "AAAA", "b": "BBBB"}, val_books={"b"})
    assert "B" not in train
    assert "AAAA" in train
    assert val["val book"] == "BBBB"


def test_the_end_of_the_transcript_is_held_out_as_before():
    # Same 90/10 split as before, so this validation set matches the one behind our old
    # numbers.
    train, val = make_splits("x" * 90 + "y" * 10, {"a": "AAAA", "b": "BBBB"}, val_books={"b"})
    assert val["val transcript"] == "y" * 10
    assert "y" not in train
    assert "x" * 90 in train
