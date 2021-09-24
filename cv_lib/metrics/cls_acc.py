from typing import List

import torch
from torch import Tensor


def accuracy(output: Tensor, target: Tensor, topk=(1,)) -> List[Tensor]:
    """
    Computes the accuracy over the k top predictions for the specified values of k

    Args:
        output: [bs, C]
        target: [bs]
    """
    with torch.no_grad():
        if target.numel() == 0:
            res = torch.zeros(len(topk)).tolist()
            return res

        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(1.0 / batch_size))
        return res
