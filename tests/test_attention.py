import torch

from pretrain.attention import AttentionHead, causal_average, causal_average_weights


def test_weights_spread_evenly_over_the_current_and_earlier_positions():
    w = causal_average_weights(3)
    expected = torch.tensor(
        [
            [1.0, 0.0, 0.0],  # position 0 sees only itself
            [1 / 2, 1 / 2, 0.0],  # position 1 averages positions 0-1
            [1 / 3, 1 / 3, 1 / 3],  # position 2 averages positions 0-2
        ]
    )
    assert torch.allclose(w, expected)


def test_causal_average_matches_a_plain_loop():
    x = torch.randn(2, 5, 4)  # (batch, time, channels)
    expected = torch.stack([x[:, : t + 1].mean(dim=1) for t in range(5)], dim=1)
    assert torch.allclose(causal_average(x), expected, atol=1e-6)


def test_the_future_cannot_leak_into_the_past():
    x = torch.randn(1, 6, 4)
    changed = x.clone()
    changed[:, 3:] = torch.randn(1, 3, 4)  # rewrite positions 3, 4, 5
    # Positions 0-2 must not notice anything that happened after them.
    assert torch.allclose(causal_average(x)[:, :3], causal_average(changed)[:, :3])


def test_head_output_has_head_size_channels_at_every_position():
    head = AttentionHead(n_embd=16, head_size=8)
    assert head(torch.randn(2, 5, 16)).shape == (2, 5, 8)


def test_head_weights_are_causal_and_each_row_sums_to_one():
    head = AttentionHead(n_embd=16, head_size=8)
    w = head.attention_weights(torch.randn(2, 5, 16))
    assert w.shape == (2, 5, 5)
    assert torch.allclose(w.sum(dim=-1), torch.ones(2, 5))
    assert torch.all(w.triu(diagonal=1) == 0)  # nothing above the diagonal


def test_head_cannot_see_the_future():
    head = AttentionHead(n_embd=16, head_size=8)
    x = torch.randn(1, 6, 16)
    changed = x.clone()
    changed[:, 3:] = torch.randn(1, 3, 16)
    with torch.no_grad():
        assert torch.allclose(head(x)[:, :3], head(changed)[:, :3])


def test_head_with_zero_queries_is_the_causal_average_of_the_values():
    # Zero queries -> every score is zero -> equal weights: back to step 10.
    head = AttentionHead(n_embd=16, head_size=8)
    with torch.no_grad():
        head.query.weight.zero_()
        x = torch.randn(2, 5, 16)
        assert torch.allclose(head(x), causal_average(head.value(x)), atol=1e-6)
