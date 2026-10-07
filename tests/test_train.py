import math

import torch

from pretrain.bigram import BigramModel
from pretrain.gpt import GPT
from pretrain.train import estimate_loss, learning_rate, loss_per_char, train_step


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


def test_gpt_can_memorize_a_single_batch():
    # The classic first check on a new model: trained over and over on one batch, it should
    # drive the loss near zero. If it can't even memorize that, something is broken.
    torch.manual_seed(0)
    model = GPT(vocab_size=20, block_size=16, n_embd=32, n_head=4, n_layer=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    x = torch.randint(20, (4, 16))
    y = torch.randint(20, (4, 16))  # random targets: nothing to learn except by memorizing
    first = train_step(model, optimizer, x, y)
    for _ in range(300):
        last = train_step(model, optimizer, x, y)
    assert first > 2.5  # starts near ln(20) = 3.0, i.e. guessing
    assert last < 0.05


def test_loss_per_character_spreads_the_token_loss_over_its_characters():
    # A token loss of 2.5 on tokens that average 2.5 characters = 1.0 per character.
    assert math.isclose(loss_per_char(2.5, n_chars=250, n_tokens=100), 1.0)


SCHEDULE = {"max_lr": 1e-3, "min_lr": 1e-4, "warmup_steps": 100, "max_steps": 1100}


def test_warmup_climbs_in_a_straight_line_to_the_peak():
    assert math.isclose(learning_rate(0, **SCHEDULE), 1e-5)  # 1/100 of the peak
    assert math.isclose(learning_rate(49, **SCHEDULE), 5e-4)  # halfway up
    assert math.isclose(learning_rate(99, **SCHEDULE), 1e-3)  # at the peak


def test_decay_follows_a_cosine_down_to_the_minimum():
    assert math.isclose(learning_rate(100, **SCHEDULE), 1e-3)  # starts at the peak
    assert math.isclose(learning_rate(600, **SCHEDULE), 5.5e-4)  # halfway: midpoint
    assert math.isclose(learning_rate(1100, **SCHEDULE), 1e-4)  # ends at the minimum


def test_learning_rate_never_leaves_the_range():
    rates = [learning_rate(step, **SCHEDULE) for step in range(1101)]
    assert all(1e-5 - 1e-12 <= lr <= 1e-3 + 1e-12 for lr in rates)
