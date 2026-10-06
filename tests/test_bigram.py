import math

import torch

from pretrain.bigram import BigramModel


def test_logits_have_one_score_per_vocabulary_entry_at_every_position():
    model = BigramModel(vocab_size=81)
    idx = torch.randint(81, (4, 8))
    logits, loss = model(idx)
    assert logits.shape == (4, 8, 81)
    assert loss is None


def test_prediction_depends_only_on_the_current_token():
    # Token 5 at the start and token 5 later on must give identical scores,
    # whatever came before it.
    model = BigramModel(vocab_size=10)
    logits, _ = model(torch.tensor([[5, 1, 2, 5]]))
    assert torch.equal(logits[0, 0], logits[0, 3])


def test_loss_is_ln_vocab_size_when_every_token_is_equally_likely():
    # With an all-zero table every token gets the same score, i.e. a uniform guess.
    model = BigramModel(vocab_size=81)
    with torch.no_grad():
        model.table.weight.zero_()
    idx = torch.randint(81, (4, 8))
    _, loss = model(idx, targets=torch.randint(81, (4, 8)))
    assert math.isclose(loss.item(), math.log(81), rel_tol=1e-6)
