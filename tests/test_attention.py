import torch

from pretrain.attention import causal_average, causal_average_weights


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
