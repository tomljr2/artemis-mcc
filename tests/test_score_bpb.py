import math

import pytest
import torch

from pretrain.score_bpb import bits_per_byte, document_nats

V = 10


def uniform_model(context, seen):
    """A model that knows nothing (equal odds on all V tokens) and records its inputs."""

    def logits_fn(x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 1 and len(x) <= context  # never shown more than its context
        seen.append(x.tolist())
        return torch.zeros(len(x), V)

    return logits_fn


@pytest.mark.parametrize("n_tokens", [1, 5, 64, 100, 1000])
@pytest.mark.parametrize("context,stride", [(8, 4), (8, 8), (64, 32), (7, 1)])
def test_every_token_is_scored_exactly_once(n_tokens, context, stride):
    # Equal odds on V tokens cost ln(V) nats per token, so the total reveals the count.
    ids = list(range(n_tokens))
    nats = document_nats([0], [i % V for i in ids], uniform_model(context, []), context, stride)
    assert nats == pytest.approx(n_tokens * math.log(V))


def test_later_windows_overlap_so_each_token_keeps_some_context():
    seen = []
    document_nats([0], [1] * 20, uniform_model(8, seen), context=8, stride=4)
    starts = [0, 4, 8, 12]  # each window starts `stride` tokens after the last one
    assert seen == [([0] + [1] * 20)[s : s + 8] for s in starts]


def test_bits_per_byte_converts_nats_to_bits_and_divides_by_bytes():
    assert bits_per_byte(2 * math.log(2), 2) == pytest.approx(1.0)
