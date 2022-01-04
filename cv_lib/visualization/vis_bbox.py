from typing import List, Optional, Union, Tuple

import cv2
import numpy as np
from PIL import Image

import torch
from torch import Tensor, LongTensor
import torchvision.transforms.functional as TF
import torchvision.ops.boxes as box_ops

from cv_lib.augmentation import UnNormalize


def draw_bbox(
    img: Union[Image.Image, np.ndarray],
    bbox_sizes: List[int],
    bbox_labels: List[int],
    color: Tuple[int] = (0, 0, 255)
):
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
        cv2.rectangle(img, p1, p2, color, 2)
        cv2.putText(
            img,
            "{}".format(bbox_label),
            org=p1,
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            lineType=cv2.LINE_AA,
            color=(255, 255, 255)
        )
    return img


def draw_img_preds(
    img: Tensor,
    bboxes: Tensor,
    bbox_labels: LongTensor,
    img_size: Tuple[int],
    color: Tuple[int] = (0, 0, 255)
) -> np.ndarray:
    """
    Args:
        img:
        bboxes: xyxy order, normalized
        img_size: (h, w)
    """
    img = TF.resize(img.squeeze(), img_size).cpu()
    img = torch.clamp(img, 0, 255).round().to(dtype=torch.uint8).permute(1, 2, 0).cpu().numpy()

    bboxes = torch.clamp(bboxes, min=0, max=1)
    if bboxes.shape[0] != 0:
        # x
        bboxes[:, [0, 2]] *= img_size[1]
        # y
        bboxes[:, [1, 3]] *= img_size[0]
    bboxes = bboxes.round().to(dtype=torch.int, device="cpu").tolist()
    bbox_labels = bbox_labels.cpu().tolist()
    img = draw_bbox(img, bboxes, bbox_labels, color=color)
    return img


def vis_bbox(
    img: Tensor,
    bboxes: Tensor,
    bbox_labels: LongTensor,
    img_size: Tuple[int],
    dataset_mean: Tuple[int],
    dataset_std: Tuple[int],
    bbox_fmt: str = "cxcywh",
    save_fp: Optional[str] = None,
    color: Tuple[int] = (0, 0, 255)
) -> np.ndarray:
    """
    Return: opencv type image
    """
    un_normalize = UnNormalize(dataset_mean, dataset_std)
    img = un_normalize(img) * 255
    bboxes = box_ops.box_convert(bboxes, bbox_fmt, "xyxy")
    img = draw_img_preds(img, bboxes, bbox_labels, img_size, color=color)
    if save_fp is not None:
        cv2.imwrite(save_fp, img)
    return img

