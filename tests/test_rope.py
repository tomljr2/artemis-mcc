import torch

from pretrain.rope import apply_rope, rope_angles


def rotate(x: torch.Tensor) -> torch.Tensor:
    """Rotate a (T, head_size) tensor, row t by the angles for position t."""
    cos, sin = rope_angles(x.shape[0], x.shape[1])
    return apply_rope(x, cos, sin)


def test_position_0_is_not_rotated():
    x = torch.randn(5, 8)
    assert torch.allclose(rotate(x)[0], x[0])


def test_rotation_keeps_every_vector_the_same_length():
    x = torch.randn(5, 8)
    assert torch.allclose(rotate(x).norm(dim=-1), x.norm(dim=-1), atol=1e-5)


def test_later_positions_are_rotated():
    x = torch.randn(5, 8)
    assert not torch.allclose(rotate(x)[1:], x[1:])


def test_query_key_score_depends_only_on_how_far_apart_they_are():
    # The whole point of RoPE: put the same query at position 6 and key at position 2, or the
    # same query at 9 and key at 5. Both are 4 apart, so they get the same score.
    q, k = torch.randn(8), torch.randn(8)
    same = torch.stack([q] * 12)
    q_rot, k_rot = rotate(same), rotate(torch.stack([k] * 12))
    assert torch.allclose(q_rot[6] @ k_rot[2], q_rot[9] @ k_rot[5], atol=1e-5)
    assert not torch.allclose(q_rot[6] @ k_rot[2], q_rot[6] @ k_rot[3], atol=1e-5)
