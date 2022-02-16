from copy import deepcopy
from typing import List, Tuple, Dict, Any, Union, Optional

from PIL.Image import Image

import torch
from torch import Tensor
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import torchvision.ops.boxes as box_op


__all__ = [
    "crop",
    "resize",
    "pad_bottom_right",
    "rotate"
]


def crop(
    img: Image,
    target: Dict[str, Any],
    region: Tuple[int]
) -> Tuple[Image, Dict[str, Any]]:
    """
    Args:
        region: [Top, Left, H, W]
    """
    # crop image
    src_w, src_h = img.size
    img = TF.crop(img, *region)

    target = deepcopy(target)
    top, left, h, w = region

    # set new image size
    if "size" in target:
        target["size"] = (h, w)

    fields: List[str] = list()
    for k, v in target.items():
        if isinstance(v, Tensor):
            fields.append(k)

    # crop bounding boxes
    if "boxes" in target:
        boxes = target["boxes"]
        boxes[:, [0, 2]] *= src_w
        boxes[:, [1, 3]] *= src_h
        boxes = box_op.box_convert(boxes, "cxcywh", "xyxy")
        boxes -= torch.tensor([left, top, left, top])
        boxes = box_op.clip_boxes_to_image(boxes, (h, w))
        keep = box_op.remove_small_boxes(boxes, 1)
        boxes[:, [0, 2]] /= w
        boxes[:, [1, 3]] /= h
        boxes = box_op.box_convert(boxes, "xyxy", "cxcywh")
        target["boxes"] = boxes
        for field in fields:
            target[field] = target[field][keep]

    if "masks" in target:
        target['masks'] = target['masks'][:, top:top + h, left:left + w]
        keep = target['masks'].flatten(1).any(1)
        for field in fields:
            target[field] = target[field][keep]

    return img, target


def resize(
    img: Image,
    target: Dict[str, Any],
    size: Union[int, Tuple[int, int]],
    max_size=None,
    interpolation_mode: TF.InterpolationMode = TF.InterpolationMode.BILINEAR,
) -> Tuple[Image, Dict[str, Any]]:
    """
    Args:
        size: [h, w]
    """

    def get_size_with_aspect_ratio(image_size, size: int, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(img.size, size, max_size)
    img = TF.resize(img, size, interpolation=interpolation_mode)

    if "size" in target:
        target["size"] = tuple(size)

    if "masks" in target:
        target['masks'] = F.interpolate(target['masks'][:, None].float(), size, mode="nearest")[:, 0] > 0.5

    return img, target


def pad_bottom_right(
    img: Image,
    target: Dict[str, Any],
    padding: Tuple[int, int]
) -> Tuple[Image, Dict[str, Any]]:
    # assumes that we only pad on the bottom right corners
    w, h = img.size
    img = TF.pad(img, (0, 0, padding[0], padding[1]))

    if "size" in target:
        target["size"] = (img.size[1], img.size[0])

    if "boxes" in target:
        bboxes = box_op.box_convert(target["boxes"], "cxcywh", "xyxy")
        x_ratio = w / (w + padding[0])
        y_ratio = h / (h + padding[1])
        bboxes *= torch.tensor([x_ratio, y_ratio, x_ratio, y_ratio])
        target["boxes"] = box_op.box_convert(bboxes, "xyxy", "cxcywh")
    if "masks" in target:
        target['masks'] = TF.pad(target['masks'], (0, 0, padding[0], padding[1]))

    return img, target


def pad(
    img: Image,
    target: Dict[str, Any],
    padding: List[int],
    fill: int = 0,
    mask_fill: int = 0,
    padding_mode: str = "constant"
) -> Tensor:
    img = TF.pad(img, padding, fill, padding_mode)

    if "size" in target:
        target["size"] = (img.size[1], img.size[0])

    if "boxes" in target:
        raise NotImplementedError
    if "masks" in target:
        target['masks'] = TF.pad(target['masks'], padding, mask_fill, padding_mode)

    return img, target


def rotate(
    img: Image,
    target: Dict[str, Any],
    angle: float,
    expand: bool = False,
    center: Optional[List[int]] = None,
    fill: Optional[List[float]] = None
) -> Tuple[Image, Dict[str, Any]]:
    img = TF.rotate(
        img,
        angle=angle,
        interpolation=TF.InterpolationMode.NEAREST,
        expand=expand,
        center=center,
        fill=fill
    )

    if "boxes" in target:
        raise NotImplementedError
    if "masks" in target:
        target["masks"] = TF.rotate(
            target["masks"],
            angle=angle,
            interpolation=TF.InterpolationMode.NEAREST,
            expand=expand,
            center=center,
            fill=fill
        )

    return img, target

