from pretrain.dataset import train_val_split


def test_split_sizes_follow_the_fraction():
    train, val = train_val_split(list(range(100)), val_fraction=0.1)
    assert len(train) == 90
    assert len(val) == 10


def test_split_keeps_order_with_validation_at_the_end():
    ids = list(range(100))
    train, val = train_val_split(ids, val_fraction=0.1)
    assert train == ids[:90]
    assert val == ids[90:]
