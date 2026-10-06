import torch

from pretrain.char_tokenizer import CharTokenizer
from pretrain.gpt import GPT
from pretrain.sample import continue_text

VOCAB = "\nEagle, Houston. Roger."


def test_output_starts_with_the_prompt_and_adds_the_requested_characters():
    torch.manual_seed(0)
    tok = CharTokenizer(VOCAB)
    model = GPT(tok.vocab_size, block_size=8, n_embd=16, n_head=4, n_layer=2).eval()
    text = continue_text(model, tok, "Eagle, Houston.", max_new_tokens=30)
    assert text.startswith("Eagle, Houston.")
    assert len(text) == len("Eagle, Houston.") + 30


def test_temperature_is_passed_to_the_model():
    tok = CharTokenizer(VOCAB)
    torch.manual_seed(0)
    model = GPT(tok.vocab_size, block_size=8, n_embd=16, n_head=4, n_layer=2).eval()
    texts = set()
    for seed in (1, 2, 3):
        torch.manual_seed(seed)
        texts.add(continue_text(model, tok, "Roger.", max_new_tokens=30, temperature=1e-4))
    assert len(texts) == 1  # near-zero temperature: same text whatever the seed
