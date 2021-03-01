import torch
from torch import Tensor, LongTensor
from torchvision.ops.boxes import box_iou


__all__ = [
    "topk_scores",
    "nms"
]


def topk_scores(scores: Tensor, topk: int, threshold: float = 0.6) -> LongTensor:
    """
    Select top k index that corresponding score is greater than threshold,
    the output maybe less than `k`
    """
    values, indices = torch.sort(scores)
    mask = values > threshold
    indices = indices[mask][:topk]
    return indices


def nms(boxes: Tensor, scores: Tensor, overlap=0.5, top_k=200):
    """
    Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.

    Args:
        boxes: (tensor) The location preds for the img, Shape: [num_priors,4].
        scores: (tensor) The class predscores for the img, Shape:[num_priors].
        overlap: (float) The overlap thresh for suppressing unnecessary boxes.
        top_k: (int) The Maximum number of box preds to consider.

    Warning this function is DIFFERENT from torchvision.ops.boxes.nms which remove
    boxes with low score first

    Return:
        The indices of the kept boxes with respect to num_priors.
    """
    device = boxes.device
    if boxes.numel() == 0:
        return torch.tensor([], dtype=torch.int64, device=device)

    _, idx = scores.sort(0, descending=True)
    idx = idx[:top_k]
    keep = list()

    while idx.shape[0] >= 1:
        new_idx = idx[0]
        keep.append(new_idx)
        selected_bbox = boxes[new_idx]
        iou = box_iou(selected_bbox.unsqueeze(0), boxes[idx])
        idx = idx[(iou < overlap).squeeze()]

    return torch.tensor(keep, dtype=torch.int64, device=device)
