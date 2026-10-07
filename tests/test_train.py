import math

import torch

from pretrain.bigram import BigramModel
from pretrain.train import estimate_loss, train_step


def test_train_step_lowers_the_loss_on_a_repeated_batch():
    torch.manual_seed(0)
    model = BigramModel(vocab_size=10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    x = torch.randint(10, (4, 8))
    y = torch.randint(10, (4, 8))
    first = train_step(model, optimizer, x, y)
    for _ in range(50):
        last = train_step(model, optimizer, x, y)
    assert last < first


def test_estimate_loss_reports_every_named_split():
    # An all-zero table guesses uniformly, so every split scores exactly ln(vocab_size).
    model = BigramModel(vocab_size=10)
    with torch.no_grad():
        model.table.weight.zero_()
    data = torch.randint(10, (200,))
    splits = {"train": data, "val transcript": data, "val book": data}
    losses = estimate_loss(model, splits, block_size=8, batch_size=4, eval_batches=3)
    assert set(losses) == set(splits)
    for loss in losses.values():
        assert math.isclose(loss, math.log(10), rel_tol=1e-6)
