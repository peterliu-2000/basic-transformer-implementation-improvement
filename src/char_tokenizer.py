import torch
class CharTokenizer:
    def __init__(self, chars):
        self.chars = chars
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, text):
        """Encodes a string into a list of integers."""
        return [self.stoi[c] for c in text]

    def decode(self, tokens):
        """Decodes a list of integers back into a string."""
        return ''.join([self.itos[i] for i in tokens])

    def debug(self):
        print(self.chars)