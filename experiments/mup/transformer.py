import math
from typing import Optional, Union, Tuple
from dataclasses import asdict, dataclass, field
import torch
import torch.nn as nn
from torch.nn import functional as F
from xformers.ops import fmha, AttentionBias
from torch.nn.attention.flex_attention import (
    BlockMask,
    flex_attention,
    _mask_mod_signature,
)

from codebase.transformer import (
    BaseTransformerArgs,
    BaseTransformer,
    TransformerBlock,
    Attention,
    FeedForward,
    flex_attention_comp,
    RMSNorm,
    cross_entropy,
    apply_rotary_emb,
    reshape_for_broadcast,
    repeat_kv,
    InitStdFactor,
)

from experiments.baseline_transformer.transformer import create_causal_mask

from codebase.optim import OptimArgs, build_lr_fn


@dataclass
class MupTransformerArgs(BaseTransformerArgs):
    seed = 42
    vocab_size = -1
    weight_tying = False
    sliding_window = None
    input_alpha = 1.0
    output_alpha = 1.0
    scaling_factor: float = 1.0


class MupAttention(Attention):
    def __init__(
        self,
        dim: int,
        head_dim: int,
        n_heads: int,
        n_kv_heads: int,
        rope_theta: float,
        scaling_factor: float,
    ):
        super().__init__(
            dim,
            head_dim,
            n_heads,
            n_kv_heads,
            rope_theta,
        )
        self.scaling_factor = scaling_factor

    def forward(
        self,
        x: torch.Tensor,
        freq_cis: torch.Tensor,
        tok_idx: Optional[torch.Tensor] = None,
        mask: Optional[Union[BlockMask, AttentionBias, str]] = None,
        attn_impl: str = "sdpa",
    ):
        # B S D
        bsz, seq_len, dim = x.shape
        xq = self.wq(x.view_as(x))
        xk = self.wk(x.view_as(x))
        xv = self.wv(x.view_as(x))

        output_shape = xq.shape
        # B S D -> B S H D
        xq = xq.view(bsz, seq_len, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, 1, freq_cis[0:seq_len])

        # This condition helps us be easily compatible
        # with inference by adding a pluggable KVCache
        if hasattr(self, "kv_cache"):
            xk, xv = self.kv_cache.update(xk, xv, tok_idx)

        xk = repeat_kv(xk, self.heads_per_group, dim=2)
        xv = repeat_kv(xv, self.heads_per_group, dim=2)

        attention_scaling_factor = 1.0 / self.n_heads

        if attn_impl == "flex_attention":
            assert mask is None or isinstance(mask, BlockMask)
            xq, xk, xv = map(lambda e: e.transpose(1, 2), (xq, xk, xv))
            output = flex_attention_comp(
                xq, xk, xv, block_mask=mask, scale=attention_scaling_factor
            )
            output = output.transpose(1, 2).contiguous()  # B H S D -> B S H D

        elif attn_impl == "fmha":
            assert mask is None or isinstance(mask, AttentionBias)
            output = fmha.memory_efficient_attention(
                xq, xk, xv, attn_bias=mask, scale=attention_scaling_factor
            )
            # This uses B S H D instead of B H S D of pytorch

        elif attn_impl == "sdpa":
            xq, xk, xv = map(lambda e: e.transpose(1, 2), (xq, xk, xv))
            assert mask is None or isinstance(mask, (str, torch.Tensor))
            is_causal = (mask == "causal") if isinstance(mask, str) else False
            mask = mask if isinstance(mask, torch.Tensor) else None
            output = F.scaled_dot_product_attention(
                xq,
                xk,
                xv,
                is_causal=is_causal,
                attn_mask=mask,
                scale=attention_scaling_factor,
            )
            output = output.transpose(1, 2).contiguous()  # B H S D -> B S H D
        else:
            raise NotImplementedError(
                f"Attention implementation {attn_impl} not supported"
            )

        output = self.wo(output.reshape(output_shape))

        return output

    def reset_parameters(self, init_std=None, out_proj_factor=1.0):
        init_std = init_std or (self.dim ** (-0.5))

        for w in [self.wq, self.wk, self.wv]:
            nn.init.trunc_normal_(
                w.weight,
                mean=0.0,
                std=init_std / math.sqrt(self.scaling_factor),
                a=-3 * init_std,
                b=3 * init_std,
            )

        nn.init.trunc_normal_(
            self.wo.weight,
            mean=0.0,
            std=init_std / out_proj_factor,
            a=-3 * init_std,
            b=3 * init_std,
        )


class MupFeedForward(FeedForward):
    def __init__(
        self,
        dim,
        hidden_dim,
        multiple_of,
        ffn_dim_multiplier,
        scaling_factor: float = 1.0,
        mp_size: int = 1,
    ):
        super().__init__(
            dim=dim,
            hidden_dim=hidden_dim,
            multiple_of=multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier,
            mp_size=mp_size,
        )
        self.scaling_factor = scaling_factor

    def reset_parameters(
        self,
        init_std=None,
        factor=1.0,
    ):
        in_init_std = init_std or (self.dim ** (-0.5))
        out_init_std = init_std or (self.hidden_dim ** (-0.5))
        in_init_std = in_init_std / math.sqrt(self.scaling_factor)
        out_init_std = out_init_std / factor
        for w in [self.w1, self.w3]:
            nn.init.trunc_normal_(
                w.weight,
                mean=0.0,
                std=in_init_std,
                a=-3 * in_init_std,
                b=3 * in_init_std,
            )
        nn.init.trunc_normal_(
            self.w2.weight,
            mean=0.0,
            std=out_init_std,
            a=-3 * out_init_std,
            b=3 * out_init_std,
        )


class MupTransformerBlock(TransformerBlock):
    def __init__(self, args):
        super().__init__(args)
        self.scaling_factor = args.scaling_factor
        self.attention = MupAttention(
            dim=args.dim,
            head_dim=self.head_dim,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            rope_theta=args.rope_theta,
            scaling_factor=self.scaling_factor,
        )
        self.feed_forward = MupFeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
            scaling_factor=args.scaling_factor,
        )


class MupTransformer(BaseTransformer):
    def __init__(
        self,
        args,
    ):
        super().__init__(args)
        self.input_alpha = args.input_alpha
        self.output_alpha = args.output_alpha
        self.weight_tying = args.weight_tying
        self.sliding_window = args.sliding_window
        self.scaling_factor = args.scaling_factor

        assert args.vocab_size > 0
        assert args.scaling_factor >= 1, "You need to set this!!!"

        self.tok_embeddings = torch.nn.Embedding(args.vocab_size, args.dim)
        # This is Post-Layer Norm layer or Post-Norm
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)

        if args.weight_tying:
            self.output = TiedLinear(self.tok_embeddings)
        else:
            self.output = nn.Linear(args.dim, args.vocab_size, bias=False)
        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(MupTransformerBlock(args))

    def forward(
        self,
        token_values: torch.Tensor,
        target=None,
        tok_idx=None,
        mask=None,
        attn_impl="sdpa",
    ):
        bsz, seqlen = token_values.shape
        mask = (
            mask
            if mask is not None
            else create_causal_mask(seqlen, attn_impl, self.sliding_window)
        )
        # (krotonus) NOTE: Embedding FWD MUP
        h = self.input_alpha * self.tok_embeddings(token_values)
        freq_cis = self.rope_embeddings(seqlen=self.max_seqlen, tok_idx=tok_idx)

        for i, layer in enumerate(self.layers):
            h = layer(h, freq_cis, tok_idx=tok_idx, mask=mask, attn_impl=attn_impl)

        # (krotonus) NOTE: Output Logit FWD. MUP
        logits = (self.output(self.norm(h)) * self.output_alpha) / self.scaling_factor
        if target is not None:
            return cross_entropy(logits, target)
        else:
            return logits
            
    def reset_parameters(self, init_std=None):
        # Either use fixed base std or sqrt model dim
        super().reset_parameters()
        init_std = init_std or (self.dim ** (-0.5))
        self.norm.reset_parameters()
        nn.init.trunc_normal_(
            self.tok_embeddings.weight,
            mean=0.0,
            std=init_std,
            a=-3 * init_std,
            b=3 * init_std,
        )
        if not self.weight_tying:
            nn.init.trunc_normal_(
                self.output.weight,
                mean=0.0,
                std=init_std,
                a=-3 * init_std,
                b=3 * init_std,
            )

    def init_weights(self, args):
        self.reset_parameters()
        out_proj_factor = math.sqrt(2 * args.model.n_layers * args.model.scaling_factor)
        for depth, layer in enumerate(self.layers):
            layer.init_weights(self.init_base_std, out_proj_factor)
