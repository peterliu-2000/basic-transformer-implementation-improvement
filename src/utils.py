import torch
import re

import re


def clean_bpe_text(text):
    # Remove all existing spaces
    text = text.replace(" ", "")
    # Convert 'G' followed by a dot to space
    text = text.replace("Ġ", " ")
    # Convert 'C' followed by a dot to newline character
    text = text.replace("Ċ", "\n")

    return text


def get_batch(data, context_window_size, device, batch_size=32):
    """
    generate a small batch of data of inputs x and targets y
    """
    #all valid context window starting idx
    ix = torch.randint(len(data) - context_window_size, (batch_size,))
    x = torch.stack([data[i:i+context_window_size] for i in ix])
    y = torch.stack([data[i+1:i+context_window_size+1] for i in ix])
    x = x.to(device)
    y = y.to(device)
    return x, y

def get_batch_tokens(encoded, context_window_size, device, batch_size=32):
    """
    generate a small batch of data of inputs x and targets y
    """
    encoded = torch.tensor(encoded, device=device, dtype=torch.long)
    ix = torch.randint(len(encoded) - context_window_size, (batch_size,))
    x = torch.stack([encoded[i:i+context_window_size] for i in ix])
    y = torch.stack([encoded[i+1:i+context_window_size+1] for i in ix])
    x = x.to(device)
    y = y.to(device)
    return x, y

def estimate_loss(model, eval_iters, context_window_size, device):
    """
    Args:
      model: model being evaluated
      eval_iters: number of batches to average over
      context_window_size: size of the context window
      device: 'cpu' or 'cuda' (should be 'cuda' if available)
    """
    out = {}
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, context_window_size, device)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    return out

import torch
import torch.nn as nn

class RotaryPositionalEmbedding:
    def __init__(self, head_dim, max_seq_len=2048, theta=10000):
        """
        Args:
            head_dim (int): Dimension of each head (should be even for RoPE).
            max_seq_len (int): Maximum sequence length.
            theta (float): Base frequency for RoPE.
        """
        assert head_dim % 2 == 0, "Head dimension must be even for RoPE"
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.device = 'mps'

        # Precompute cos/sin embeddings
        self._build_freqs()

    def _build_freqs(self):
        """Precompute the cos and sin position embeddings."""
        dim = self.head_dim // 2  # RoPE operates on pairs of elements
        positions = torch.arange(self.max_seq_len).float().unsqueeze(1).to(self.device)  # (T, 1)
        indices = torch.arange(dim).float().to(self.device)  # (K/2,)
        theta_inv = 1.0 / (self.theta ** (indices / dim))  # (K/2,)

        # Compute angles
        angle = positions * theta_inv  # (T, K/2)

        # Compute cos and sin embeddings
        self.cos_emb = angle.cos().unsqueeze(0).unsqueeze(0)  # (B, 1, T, K/2)
        self.sin_emb = angle.sin().unsqueeze(0).unsqueeze(0)  # (B, 1, T, K/2)

    def rotate_queries_or_keys(self, x):
        """
        Applies RoPE rotation to queries or keys.

        Args:
            x (Tensor): (B, 1, T, K) tensor (queries or keys)

        Returns:
            Tensor: Rotated queries/keys (B, 1, T, D)
        """
        B, _, T, K = x.shape
        assert K == self.head_dim, f"Head dim mismatch: expected {self.head_dim}, got {D}"

        # Ensure cos and sin embeddings are properly shaped
        cos_emb = self.cos_emb[:, :, :T, :].to(self.device)  # (1, 1, T, K/2)
        sin_emb = self.sin_emb[:, :, :T, :].to(self.device)  # (1, 1, T, K/2)

        # Split x into real and imaginary parts
        x_1, x_2 = x[..., :K//2], x[..., K//2:]  # (B, 1, T, D/2)

        # Apply rotation
        rotated_x = torch.cat([x_1 * cos_emb - x_2 * sin_emb,
                               x_2 * cos_emb + x_1 * sin_emb], dim=-1).to(self.device)

        return rotated_x  # (B, 1, T, K)


