import torch

from pretrain.checkpoint import load_checkpoint, save_checkpoint
from pretrain.gpt import GPT

MODEL_ARGS = {"vocab_size": 20, "block_size": 8, "n_embd": 16, "n_head": 4, "n_layer": 2}


def test_a_loaded_model_predicts_exactly_like_the_saved_one(tmp_path):
    torch.manual_seed(0)
    model = GPT(**MODEL_ARGS).eval()
    path = tmp_path / "model.pt"
    save_checkpoint(path, model, MODEL_ARGS, vocab="abcdefghijklmnopqrst", step=7, val_loss=1.5)

    loaded, vocab, info = load_checkpoint(path)
    idx = torch.randint(20, (2, 8))
    with torch.no_grad():
        assert torch.equal(loaded(idx)[0], model(idx)[0])
    assert vocab == "abcdefghijklmnopqrst"
    assert info == {"step": 7, "val_loss": 1.5}


def test_loaded_model_is_ready_for_generation(tmp_path):
    # Loading should hand back a model in eval mode, so dropout is off when sampling.
    model = GPT(**MODEL_ARGS, dropout=0.5)
    path = tmp_path / "model.pt"
    save_checkpoint(path, model, {**MODEL_ARGS, "dropout": 0.5}, "x" * 20, step=0, val_loss=9.9)
    loaded, _, _ = load_checkpoint(path)
    assert not loaded.training
