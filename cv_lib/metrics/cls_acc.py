from typing import List

import torch


def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)) -> List[torch.Tensor]:
    """
    Computes the accuracy over the k top predictions for the specified values of k

    Args:
        output: [bs, C]
        target: [bs]
    """
    device = output.device
    with torch.no_grad():
        if target.numel() == 0:
            res = [torch.tensor(0, dtype=torch.float, device=device) for _ in topk]
            return res

        maxk = max(topk)
        batch_size = target.shape[0]

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred)).float()

        res = []
        for k in topk:
            correct_k = correct[:k].sum()
            res.append(correct_k / batch_size)
        return res
