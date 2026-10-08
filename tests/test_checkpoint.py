import torch

from pretrain.bpe import BPETokenizer
from pretrain.checkpoint import load_checkpoint, save_checkpoint
from pretrain.gpt import GPT

TOKENIZER = BPETokenizer.train("Houston, Tranquility Base here. " * 5, vocab_size=270)
MODEL_ARGS = {"vocab_size": 270, "block_size": 8, "n_embd": 16, "n_head": 4, "n_layer": 2}


def test_a_loaded_model_predicts_exactly_like_the_saved_one(tmp_path):
    torch.manual_seed(0)
    model = GPT(**MODEL_ARGS).eval()
    path = tmp_path / "model.pt"
    save_checkpoint(path, model, MODEL_ARGS, TOKENIZER, step=7, val_loss=1.5)

    loaded, _, info = load_checkpoint(path)
    idx = torch.randint(270, (2, 8))
    with torch.no_grad():
        assert torch.equal(loaded(idx)[0], model(idx)[0])
    assert info == {"step": 7, "val_loss": 1.5}


def test_the_tokenizer_travels_with_the_model(tmp_path):
    path = tmp_path / "model.pt"
    save_checkpoint(path, GPT(**MODEL_ARGS), MODEL_ARGS, TOKENIZER, step=0, val_loss=9.9)
    _, tok, _ = load_checkpoint(path)
    assert list(tok.merges.items()) == list(TOKENIZER.merges.items())
    assert tok.pattern == TOKENIZER.pattern  # and the rule its merges were learned with


def test_loaded_model_is_ready_for_generation(tmp_path):
    # Loading should hand back a model in eval mode, so dropout is off when sampling.
    model = GPT(**MODEL_ARGS, dropout=0.5)
    path = tmp_path / "model.pt"
    save_checkpoint(path, model, {**MODEL_ARGS, "dropout": 0.5}, TOKENIZER, step=0, val_loss=9.9)
    loaded, _, _ = load_checkpoint(path)
    assert not loaded.training
