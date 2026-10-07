import torch

from pretrain.bpe import BPETokenizer
from pretrain.gpt import GPT
from pretrain.sample import continue_text

TOKENIZER = BPETokenizer.train("\nEagle, Houston. Roger. " * 10, vocab_size=280)


def test_output_starts_with_the_prompt_and_adds_the_requested_tokens():
    torch.manual_seed(0)
    tok = TOKENIZER
    model = GPT(tok.vocab_size, block_size=8, n_embd=16, n_head=4, n_layer=2).eval()
    text = continue_text(model, tok, "Eagle, Houston.", max_new_tokens=30)
    assert text.startswith("Eagle, Houston.")
    # Each token spells at least one byte, so 30 new tokens add at least 30 bytes.
    assert len(text.encode("utf-8")) >= len("Eagle, Houston.") + 30


def test_temperature_is_passed_to_the_model():
    tok = TOKENIZER
    torch.manual_seed(0)
    model = GPT(tok.vocab_size, block_size=8, n_embd=16, n_head=4, n_layer=2).eval()
    texts = set()
    for seed in (1, 2, 3):
        torch.manual_seed(seed)
        texts.add(continue_text(model, tok, "Roger.", max_new_tokens=30, temperature=1e-4))
    assert len(texts) == 1  # near-zero temperature: same text whatever the seed
