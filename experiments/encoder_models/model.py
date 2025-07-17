from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from torch import nn


@dataclass
class PostModernBertArgs:
    vocab_size: int = 256
    dim: int = 512
    n_layers: int = 8
    head_dim: Optional[int] = None

    # Embedding Related Params
    pad_token_id: int = 255
    norm_eps: float = 1e-5
    norm_bias: float = False
    embedding_dropout: float = 0.0


class PostModernBertMLP(nn.Module):
    def __init__(self, args):
        super().__init__()

    def forward(self, hidden_state):
        pass


class PostModernBertEncoderBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        # ModernBERT uses nn.Identity as attn_norm for layer_idx ==0
        self.attn_norm = nn.LayerNorm(args.dim, eps=args.norm_eps, bias=args.norm_bias)
        self.attn = PostModernAttention(args)
        self.mlp = PostModernBertMLP(args)
        self.mlp_norm = nn.LayerNorm(args.dim, eps=args.norm_eps, bias=args.norm_bias)

    def forward(self, hidden_states):
        attn_outputs = self.attn()
        # Residual Connection
        hidden_states = hidden_states + attn_outputs
        # MLP Output
        mlp_output = self.mlp(self.mlp_norm(hidden_states))
        return inp


class PostModernEmbeddings(nn.Module):
    """
    Currently similar to ModernBert without `torch.compile`
    """

    def __init__(self, args):
        super().__init__()
        self.tok_embeddings = torch.nn.Embedding(
            args.vocab_size, args.dim, padding_idx=args.pad_token_id
        )
        self.norm = nn.LayerNorm(args.dim, eps=args.norm_eps, bias=args.norm_bias)
        self.drop = nn.Dropout(args.embedding_dropout)

    def forward(self, input_ids):
        h = self.drop(self.norm(self.tok_embeddings(input_ids)))
        return h


class PostModernBert(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.embeddings = PostModernEmbeddings(args)
        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(EncoderBlock(args))
        self.pooler = None
        self.final_norm = None

    def forward(self, inp):
        h = self.embeddings(inp)
        for i, layer in enumerate(self.layers):
            h = layer(h)
        return h
