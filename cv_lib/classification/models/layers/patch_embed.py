from copy import deepcopy
from typing import Union, Tuple

import torch
import torch.nn as nn

from .helper import to_2tuple


class PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size: Union[Tuple[int, int], int],
        image_channels: int,
        embed_dim: int,
    ):
        super().__init__()
        self.img_size = to_2tuple(img_size)
        self.image_channels = image_channels
        self.embed_dim = embed_dim

        self.grid_size: Tuple[int, int] = None
        self.num_patches: int = None


class ViTPatchEmbed(PatchEmbed):
    """
    2D Image to Patch Embedding
    """
    def __init__(
        self,
        img_size: Union[Tuple[int, int], int] = 224,
        patch_size: Union[Tuple[int, int], int] = 16,
        image_channels: int = 3,
        embed_dim: int = 768,
        **kwargs
    ):
        super().__init__(img_size=img_size, image_channels=image_channels, embed_dim=embed_dim)
        self.patch_size = to_2tuple(patch_size)

        self.proj = nn.Conv2d(image_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.flatten = nn.Flatten(2)

        self.grid_size = (self.img_size[0] // self.patch_size[0], self.img_size[1] // self.patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor):
        x = self.proj(x)
        # [bs, C, H, W] -> [bs, C, N]
        x = self.flatten(x)
        # [bs, C, N] -> [N, bs, C]
        x = x.permute(2, 0, 1)
        return x


__REGISTERED_PATCH_EMBED__ = {
    "vit_like": ViTPatchEmbed,
}


def get_patch_embedding(embed_dim: int, **patch_embed_cfg) -> PatchEmbed:
    patch_embed_cfg = deepcopy(patch_embed_cfg)
    name = patch_embed_cfg.pop("name")
    pos_embed = __REGISTERED_PATCH_EMBED__[name](embed_dim=embed_dim, **patch_embed_cfg)
    return pos_embed
