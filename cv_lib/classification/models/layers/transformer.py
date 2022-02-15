from typing import Optional, Callable

import torch.nn as nn
from torch import Tensor


class Transformer(nn.Module):
    def __init__(
        self,
        encoder_layer_gen: Callable[[], nn.Module],
        depth: int = 12,
        embed_dim: int = 512,
        final_norm: bool = True,
        norm_eps: float = 1.0e-5,
        pre_norm: bool = True
    ):
        super().__init__()
        self.depth = depth
        self.embed_dim = embed_dim
        self.pre_norm = pre_norm

        self.norm = nn.LayerNorm(embed_dim, eps=norm_eps) if final_norm else None
        self.blocks = nn.ModuleList([encoder_layer_gen() for _ in range(depth)])

    def pre_forward(
        self,
        seq: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None
    ):
        for block in self.blocks:
            seq = block(
                seq,
                src_mask=src_mask,
                src_key_padding_mask=src_key_padding_mask
            )
        if self.norm is not None:
            seq = self.norm(seq)
        return seq

    def post_forward(
        self,
        seq: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None
    ):
        if self.norm is not None:
            seq = self.norm(seq)
        for block in self.blocks:
            seq = block(
                seq,
                src_mask=src_mask,
                src_key_padding_mask=src_key_padding_mask
            )
        return seq

    def forward(
        self,
        seq: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None
    ):
        if self.pre_norm:
            forward_fn = self.pre_forward
        else:
            forward_fn = self.post_forward
        return forward_fn(seq, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)
