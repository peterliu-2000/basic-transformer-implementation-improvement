import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils import *

device = 'mps'

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size, context_window_size, embed_size=384):
        """
        Args:
          head_size: int, size of the head embedding dimension (K)
          context_window_size: int, number of tokens considered in the past for attention (T)
          embed_size: int, size of the token embedding dimension (D)

        Uk: D,K
        Uq: D,K
        Uv: D,D
        """
        super().__init__()
        self.head_size = head_size
        # Need to explicitly turn off the biase term
        # Notice it's Linear here, which applies transformation from K to D
        self.key = nn.Linear(embed_size, head_size, bias=False)
        self.query = nn.Linear(embed_size, head_size, bias=False)
        self.value = nn.Linear(embed_size, embed_size, bias=False)
        self.embed_size = embed_size

        # not a param of the model, so registered as a buffer so will not be learned is
        self.register_buffer('tril', torch.tril(
            torch.ones(context_window_size, context_window_size)))

    def forward(self, x):
        """
        Args:
          x: (B,T,D) tensor of token embeddings

        Returns:
          (B,T,D) tensor of attention-weighted token embeddings
        """

        B, T, D = x.shape
        q = self.query(x)  # B, T, K
        k = self.key(x)  # B, T, K
        v = self.value(x)  # B, T, D
        positions = torch.arange(T, dtype=torch.long, device=x.device)

        q = q.view(B, 1, T, self.head_size)
        k = k.view(B, 1, T, self.head_size)
        v = v.view(B, 1, T, self.embed_size)

        # Apply RoPE to q and k
        rotary_embed = RotaryPositionalEmbedding(self.head_size)
        q = rotary_embed.rotate_queries_or_keys(q)  # (B, 1, T, head_size)
        k = rotary_embed.rotate_queries_or_keys(k)  # (B, 1, T, head_size)

        attention_scores = torch.matmul(q, k.transpose(-2, -1))  # B, 1, T, T
        attention_scores /= self.head_size ** 0.5
        masked_attention = attention_scores.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # B, T, T
        attention_weights = F.softmax(masked_attention, dim=-1)  # B, T, T
        output = torch.matmul(attention_weights, v)  # B, 1, T, head_size

        return output # B, 1, T, head_size

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, context_window_size, num_heads, embed_size=384):
        """
        Args:
            context_window_size: int, number of tokens considered in the past for attention (T)
            num_heads: int, number of heads (H)
            head_size: int, size of the head embedding dimension (K)
            embed_size: int, size of the token embedding dimension
        """
        super().__init__()
        # TODO, your code below
        self.head_size = embed_size // num_heads
        self.heads = nn.ModuleList([
            Head(self.head_size, context_window_size, embed_size) for _ in range(num_heads)
        ])
        self.proj = nn.Linear(num_heads * embed_size, embed_size)

    def forward(self, x):
        """
        Args:
            x: B, T, D: tensors of token embeddings
        """

        head_outputs = [head(x) for head in self.heads]  # length H list of B, 1, T, head_size
        output = torch.concat(head_outputs, dim=1) #B, H, T, head_size
        B, H, T, head_size = output.shape
        output = output.permute(0, 2, 1, 3).reshape(B, T, H * head_size)
        output = self.proj(output)  # B, T, D
        return output

class FeedForward(nn.Module):
    """
    This is applied after the multi-headed attention layer to introduce non-linearity and increase model
    capacity
    """

    def __init__(self, embed_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.ReLU(),
            nn.Linear(4 * embed_size, embed_size),
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    """
        Upgraded version of TransformerBlock where dropout layer is introduced after layernorm of attention and
    feedforward layer
    """

    def __init__(self, vocab_size, context_window_size, embed_size=384, num_heads=6, dropout=0.2):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_size)
        self.ln2 = nn.LayerNorm(embed_size)
        self.context_window_size = context_window_size
        self.dropout = nn.Dropout(dropout)
        self.feed_forward = FeedForward(embed_size)
        self.atten_heads = MultiHeadAttention(context_window_size, embed_size=embed_size, num_heads=num_heads)

    def forward(self, x):
        atten_output = self.atten_heads(self.ln1(x))
        x = x + self.dropout(atten_output)
        # non-linearity introduced here
        ff_output = self.feed_forward(self.ln2(x))
        x = x + self.dropout(ff_output)
        return x


class TransformerUpgraded(nn.Module):

    def __init__(self, vocab_size, context_window_size, embed_size=384, num_heads=6, n_layers=6):
        """
          Args:
              vocab_size: int, number of tokens in the vocabulary (V)
              context_window_size: int, size of the context window (T)
              embed_size: int, embedding size (D)
              num_heads: int, number of heads (H)
              n_layers: int, number of layers (M)
        """
        super().__init__()
        self.context_window_size = context_window_size
        self.token_embedding_table = nn.Embedding(vocab_size, embed_size)
        self.blocks = nn.Sequential(*[
            TransformerBlock(vocab_size=vocab_size,
                             context_window_size=context_window_size,
                             embed_size=embed_size,
                             num_heads=num_heads)
            for _ in range(n_layers)])

        # final layer norm
        self.ln_f = nn.LayerNorm(embed_size)
        self.lm_head = nn.Linear(embed_size, vocab_size)

        # good initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, token_ids, targets=None):
        """
        Args:
            token_ids: tensor of integers, provides the contet, shape (B, T)
            targets: tensor of integers, provides the tokens we are preidcitng, shape (B, T)
        """
        B, T = token_ids.shape

        # token_ids and targets are both (B, T) tensor of integers
        x = self.token_embedding_table(token_ids)  # (B, T, D)
        x = self.blocks(x)  # B, T, D
        x = self.ln_f(x)  # B, T, D
        logits = self.lm_head(x)  # B, T, V

        loss = None

        if targets is not None:
            logits = logits.view(B * T, -1)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
            logits = logits.view(B, T, -1)

        return logits, loss

    @torch.no_grad()
    def generate(self, token_ids, max_new_tokens):
        """
        Args:
            token_ids: (B, T) tensor of token ids to provide as context
            max_new_tokens: int, maximum number of new tokens to generate

        Returns:
            (B, T + max_new_tokens) tensor of context with new tokens appended
        """
        B, T = token_ids.shape
        NLL = torch.zeros((B, 1), device=device)
        for _ in range(max_new_tokens):
            # Crop to the last context_window_size tokens
            current_token_ids = token_ids[:, -self.context_window_size:]  # (B, T)
            logits, _ = self(current_token_ids)  # (B, T, V)
            logits = logits[:, -1, :]  # (B, V)
            probs = F.softmax(logits, dim=-1)  # (B, V)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
            generated_prob = torch.gather(probs, 1, next_token)  # B, 1
            NLL -= torch.log(generated_prob).sum()
            token_ids = torch.cat((token_ids, next_token), dim=-1)  # (B, T + 1)

        NLL /= B * max_new_tokens
        perplexity = torch.exp(NLL)

        return token_ids, perplexity

