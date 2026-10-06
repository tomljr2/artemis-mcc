import torch

from pretrain.gpt import GPT


def make_model() -> GPT:
    torch.manual_seed(0)
    return GPT(vocab_size=20, block_size=8, n_embd=16, n_head=4)


def test_logits_have_one_score_per_vocabulary_entry_at_every_position():
    logits, loss = make_model()(torch.randint(20, (4, 8)))
    assert logits.shape == (4, 8, 20)
    assert loss is None


def test_loss_is_returned_when_targets_are_given():
    _, loss = make_model()(torch.randint(20, (4, 8)), torch.randint(20, (4, 8)))
    assert loss.ndim == 0  # a single number


def test_same_token_at_different_positions_gets_different_predictions():
    # Unlike the bigram model, position and context now matter.
    logits, _ = make_model()(torch.tensor([[5, 5, 5]]))
    assert not torch.allclose(logits[0, 0], logits[0, 2])


def test_predictions_cannot_see_the_future():
    model = make_model()
    idx = torch.tensor([[1, 2, 3, 4, 5, 6]])
    changed = torch.tensor([[1, 2, 3, 9, 9, 9]])
    with torch.no_grad():
        assert torch.allclose(model(idx)[0][:, :3], model(changed)[0][:, :3])


def test_generate_can_run_past_the_context_length():
    # The position table has only block_size rows, so generate must crop its context.
    out = make_model().generate(torch.tensor([[1]]), max_new_tokens=20)
    assert out.shape == (1, 21)


def test_layers_that_output_zero_pass_the_input_straight_through():
    # Zero the last layer of attention and of the MLP, so both output exactly 0. With
    # residual connections (x = x + layer(x)) the embeddings then reach the final norm and
    # lm_head unchanged.
    model = make_model()
    with torch.no_grad():
        for layer in (model.attention.proj, model.mlp.net[-1]):
            layer.weight.zero_()
            layer.bias.zero_()
        idx = torch.tensor([[1, 2, 3]])
        embeddings = model.token_embedding(idx) + model.position_embedding(torch.arange(3))
        logits, _ = model(idx)
        assert torch.allclose(logits, model.lm_head(model.ln_f(embeddings)), atol=1e-6)
