import math
from pathlib import Path

import torch
from torch import nn

from pretrain.bigram import BigramModel
from pretrain.bpe import BPETokenizer
from pretrain.gpt import GPT
from pretrain.train import (
    CONFIGS,
    TOKENIZER_PATH,
    clip_gradients,
    encode_to_tensor,
    estimate_loss,
    learning_rate,
    loss_per_char,
    parse_args,
    recipe,
    train_step,
)


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


def make_grads(scale: float) -> nn.Module:
    """A small layer whose gradients are filled in by hand, then multiplied by scale."""
    torch.manual_seed(0)
    layer = nn.Linear(4, 3)
    for p in layer.parameters():
        p.grad = torch.randn_like(p) * scale
    return layer


def grads(layer: nn.Module) -> torch.Tensor:
    return torch.cat([p.grad.flatten() for p in layer.parameters()])


def test_small_gradients_are_left_alone():
    layer = make_grads(scale=0.01)
    before = grads(layer).clone()
    clip_gradients(layer, max_norm=1.0)
    assert torch.equal(grads(layer), before)


def test_big_gradients_shrink_to_the_limit_but_keep_their_direction():
    layer = make_grads(scale=100.0)
    before = grads(layer).clone()
    norm = clip_gradients(layer, max_norm=1.0)
    assert math.isclose(norm, before.norm().item(), rel_tol=1e-5)  # reports the size before
    assert math.isclose(grads(layer).norm().item(), 1.0, rel_tol=1e-5)  # now exactly the limit
    assert torch.allclose(grads(layer) / grads(layer).norm(), before / before.norm())


def test_matches_pytorchs_built_in_clipping():
    ours, theirs = make_grads(scale=100.0), make_grads(scale=100.0)
    clip_gradients(ours, max_norm=1.0)
    nn.utils.clip_grad_norm_(theirs.parameters(), max_norm=1.0)
    assert torch.allclose(grads(ours), grads(theirs))


def test_train_step_clips_the_gradients_it_steps_with():
    torch.manual_seed(0)
    model = GPT(vocab_size=20, block_size=16, n_embd=32, n_head=4, n_layer=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x, y = torch.randint(20, (4, 16)), torch.randint(20, (4, 16))
    train_step(model, optimizer, x, y, max_grad_norm=0.01)
    # The gradients used by the step are still attached to the parameters afterwards.
    norm = torch.sqrt(sum((p.grad**2).sum() for p in model.parameters()))
    assert norm <= 0.01 + 1e-6


def test_encoding_into_compact_storage_gives_the_same_ids():
    tok = BPETokenizer.train("Houston, Tranquility Base here. 04 03 29\n" * 20, vocab_size=300)
    text = "Tranquility Base, Houston.\n\n04 03 29 CDR We're GO, café."
    ids = encode_to_tensor(tok, text)
    assert ids.dtype == torch.int16  # 2 bytes per token instead of 8
    assert ids.tolist() == tok.encode(text)


def gpt_from_config(name: str) -> GPT:
    cfg = CONFIGS[name]
    keys = ("block_size", "n_embd", "n_head", "n_layer", "dropout")
    return GPT(vocab_size=1024, **{k: cfg[k] for k in keys})


def test_every_gpt_config_builds_a_model_that_runs():
    for name in CONFIGS:
        if name.startswith("gpt"):
            model = gpt_from_config(name)
            logits, _ = model(torch.zeros((1, 8), dtype=torch.long))
            assert logits.shape == (1, 8, 1024), name


def test_the_bigger_config_has_about_ten_times_the_parameters():
    def count(name):
        return sum(p.numel() for p in gpt_from_config(name).parameters())

    assert 9 < count("gpt-11m") / count("gpt") < 13


def test_the_defaults_are_the_best_recipe_so_far():
    args = parse_args([])
    assert args.model == "gpt-11m"
    assert args.web
    assert args.nasa_weight == 0.25
    assert args.tokenizer == TOKENIZER_PATH == Path("checkpoints/tokenizer_4096_mix.json")


def test_the_11m_recipe_reads_long_windows_but_stops_before_memorizing_nasa():
    cfg = CONFIGS["gpt-11m"]
    assert cfg["block_size"] == 256
    assert cfg["block_size"] * cfg["batch_size"] == 16_384  # tokens per step, as nanochat
    assert cfg["max_steps"] == 6000  # NASA val was best here in the 40,000-step run


def test_options_override_the_recipe_and_leave_the_rest():
    cfg = recipe("gpt-11m", block_size=512, batch_size=32, max_steps=None)
    assert (cfg["block_size"], cfg["batch_size"]) == (512, 32)
    assert cfg["max_steps"] == CONFIGS["gpt-11m"]["max_steps"]  # None: not overridden
    assert CONFIGS["gpt-11m"]["block_size"] == 256  # the recipe itself is unchanged


def test_earlier_recipes_can_still_be_reproduced():
    args = parse_args(
        ["--model", "gpt", "--no-web", "--tokenizer", "checkpoints/tokenizer_1024.json"]
    )
    assert (args.model, args.web) == ("gpt", False)
    assert args.tokenizer == Path("checkpoints/tokenizer_1024.json")
