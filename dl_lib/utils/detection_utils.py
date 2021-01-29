from typing import List, Union, Tuple

import cv2
import numpy as np
from PIL import Image

import torch
from torch import Tensor, LongTensor
import torchvision.transforms.functional as TF
from torchvision.ops.boxes import box_iou


__all__ = [
    "topk_scores",
    "draw_bbox",
    "draw_img_preds",
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


def draw_img_preds(img: Tensor, bboxes: Tensor, bbox_labels: LongTensor, img_size: Tuple[int]):
    """
    Args:
        bboxes: xyxy order, normalized
        img_size: (h, w)
    """
    img = TF.resize(img.squeeze(), img_size).cpu()
    img = torch.clamp(img, 0, 255).round().to(dtype=torch.uint8).permute(1, 2, 0).cpu().numpy()[..., ::-1]

    bboxes = torch.clamp(bboxes, min=0, max=1)
    if bboxes.shape[0] != 0:
        # x
        bboxes[:, [0, 2]] *= img_size[1]
        # y
        bboxes[:, [1, 3]] *= img_size[0]
    bboxes = bboxes.round().to(dtype=torch.int, device="cpu").tolist()
    bbox_labels = bbox_labels.cpu().tolist()
    img = draw_bbox(img, bboxes, bbox_labels)
    return img


def draw_bbox(img: Union[Image.Image, np.ndarray], bbox_sizes: List[int], bbox_labels: List[int]):
    """
    Draw bounding boxes to image
    Args:
        img: pillow style image
        bbox_sizes: bounding box (x1, y1, x2, y2)
        bbox_labels: labels
    Return:
        opencv type image
    """
    if isinstance(img, Image.Image):
        img = np.array(img)

    # convert to opencv form
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    for (x1, y1, x2, y2), bbox_label in zip(bbox_sizes, bbox_labels):
        p1 = (x1, y1)
        p2 = (x2, y2)
        cv2.rectangle(img, p1, p2, (0, 0, 255), 2)
        cv2.putText(
            img,
            "{}".format(bbox_label),
            org=p1,
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            lineType=cv2.LINE_AA,
            color=(255, 255, 255)
        )
    return img
