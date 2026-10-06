"""Character-level tokenizer: every unique character in the text is one token.

A neural network only does arithmetic, so text has to become numbers before it goes in.
This is the simplest possible scheme: number the characters, one id per character.
"""


class CharTokenizer:
    def __init__(self, text: str):
        chars = sorted(set(text))  # sorted so the same text always gives the same ids
        self.vocab_size = len(chars)
        self.char_to_id = {ch: i for i, ch in enumerate(chars)}
        self.id_to_char = {i: ch for i, ch in enumerate(chars)}

    def encode(self, text: str) -> list[int]:
        """Text -> list of token ids."""
        missing = set(text) - self.char_to_id.keys()
        if missing:
            raise ValueError(f"characters not in vocabulary: {sorted(missing)}")
        return [self.char_to_id[ch] for ch in text]

    def decode(self, ids: list[int]) -> str:
        """List of token ids -> text."""
        return "".join(self.id_to_char[i] for i in ids)
