from typing import Tuple
from math import sqrt

import torch
from torchvision.utils import make_grid
import torchvision.transforms.functional as TF


__all__ = [
    "vis_featuremap",
    "vis_seq_token"
]


def vis_featuremap(
    feat: torch.Tensor,
    n_sigma: float = 3,
    padding: int = 1,
    scale_each: bool = False,
    save_fp: str = None,
    n_row: int = None,
    **grid_kwargs
) -> torch.Tensor:
    """
    Visualize featuremap of CNN

    Args:
        feat: input featuremap with shape [C, W, H]
        n_sigma: clamp outliners as Gaussian distribution to +/- n_sigma
        save_fp: filepath for save

    Return: `Tensor` with shape [3, W', H']
    """
    assert n_sigma > 0
    # add 0 dim
    feat = feat.unsqueeze(1)
    # normalize x to normal distribution
    feat = (feat - feat.mean()) / feat.std()
    # clip to +/- n_sigma
    feat.clamp_(-n_sigma, n_sigma)
    if n_row is None:
        n_row = int(sqrt(feat.shape[0]) + 0.5)
    gird_img = make_grid(
        tensor=feat,
        nrow=n_row,
        padding=padding,
        normalize=True,
        scale_each=scale_each,
        **grid_kwargs
    )
    if save_fp is not None:
        gird_img_ = TF.to_pil_image(gird_img)
        gird_img_.save(save_fp)
    return gird_img


def vis_seq_token(
    seq: torch.Tensor,
    feat_shape: Tuple[int],
    n_sigma: float = 3,
    padding: int = 1,
    scale_each: bool = False,
    save_fp: str = None,
    n_row: int = None,
    **vis_kwargs
) -> torch.Tensor:
    """
    Visualize sequence of tokens of Transformer

    Args:
        seq: input sequence of tokens with shape [N, dim]
        feat_shape: shape of image corresponding to sequence
        n_sigma: clamp outliners as Gaussian distribution to +/- n_sigma
        save_fp: filepath for save

    Return: `Tensor` with shape [3, W', H']
    """
    # make seq to [dim, W, H]
    seq = seq.permute(1, 0).unflatten(dim=-1, sizes=feat_shape)
    res = vis_featuremap(
        seq,
        n_sigma,
        padding,
        scale_each,
        save_fp,
        n_row=n_row,
        **vis_kwargs
    )
    return res

