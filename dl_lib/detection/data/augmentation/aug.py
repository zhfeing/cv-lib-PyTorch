import random
from typing import Tuple
import abc

from PIL.Image import Image

import torch
from torch import FloatTensor, LongTensor
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision.ops.boxes import box_iou


__all__ = [
    "BaseTransform",
    "Compose",
    "RandomCrop",
    "RandomHorizontalFlip",
    "ColorJitter",
    "UnNormalize"
]


class BaseTransform(abc.ABC):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def __call__(
        self,
        img: Image,
        bbox_sizes: FloatTensor,
        bbox_labels: LongTensor
    ) -> Tuple[Image, FloatTensor, LongTensor]:
        pass


class Compose(BaseTransform):
    def __init__(self, *transforms):
        super().__init__()
        self.transforms = transforms

    def __call__(self, *pack):
        for tf in self.transforms:
            pack = tf(*pack)
        return pack


class RandomCrop(BaseTransform):
    """
    Cropping for SSD, according to original paper
        Choose between following 3 conditions:
        1. Preserve the original image
        2. Random crop minimum IoU is among 0.1, 0.3, 0.5, 0.7, 0.9
        3. Random crop
        Reference to https://github.com/chauhan-utk/src.DomainAdaptation
    """
    def __init__(self):
        super().__init__()
        self.sample_options = (
            # Do nothing
            None,
            # min IoU, max IoU
            (0.1, None),
            (0.3, None),
            (0.5, None),
            (0.7, None),
            (0.9, None),
            # no IoU requirements
            (None, None),
        )

    def __call__(self, img: Image, bbox_sizes: FloatTensor, bbox_labels: LongTensor):
        # Ensure always return cropped image
        while True:
            mode = random.choice(self.sample_options)

            if mode is None:
                return img, bbox_sizes, bbox_labels

            img_w, img_h = img.size

            min_iou, max_iou = mode
            min_iou = float("-inf") if min_iou is None else min_iou
            max_iou = float("+inf") if max_iou is None else max_iou

            # Implementation use 50 iteration to find possible candidate
            for _ in range(1):
                # size of each sampled path in [0.1, 1] 0.3*0.3 approx. 0.1
                w = random.uniform(0.3, 1.0)
                h = random.uniform(0.3, 1.0)

                if w / h < 0.5 or w / h > 2:
                    continue

                # left 0 ~ wtot - w, top 0 ~ htot - h
                left = random.uniform(0, 1.0 - w)
                top = random.uniform(0, 1.0 - h)

                right = left + w
                bottom = top + h

                ious = box_iou(bbox_sizes, torch.tensor([[left, top, right, bottom]]))

                # tailor all the bboxes and return
                if not ((ious > min_iou) & (ious < max_iou)).all():
                    continue

                # discard any bboxes whose center not in the cropped image
                xc = 0.5 * (bbox_sizes[:, 0] + bbox_sizes[:, 2])
                yc = 0.5 * (bbox_sizes[:, 1] + bbox_sizes[:, 3])

                masks = (xc > left) & (xc < right) & (yc > top) & (yc < bottom)

                # if no such boxes, continue searching again
                if not masks.any():
                    continue

                bbox_sizes[bbox_sizes[:, 0] < left, 0] = left
                bbox_sizes[bbox_sizes[:, 1] < top, 1] = top
                bbox_sizes[bbox_sizes[:, 2] > right, 2] = right
                bbox_sizes[bbox_sizes[:, 3] > bottom, 3] = bottom

                bbox_sizes = bbox_sizes[masks, :]
                bbox_labels = bbox_labels[masks]

                left_idx = int(left * img_w)
                top_idx = int(top * img_h)
                right_idx = int(right * img_w)
                bottom_idx = int(bottom * img_h)

                img = img.crop((left_idx, top_idx, right_idx, bottom_idx))

                bbox_sizes[:, 0] = (bbox_sizes[:, 0] - left) / w
                bbox_sizes[:, 1] = (bbox_sizes[:, 1] - top) / h
                bbox_sizes[:, 2] = (bbox_sizes[:, 2] - left) / w
                bbox_sizes[:, 3] = (bbox_sizes[:, 3] - top) / h

                img_h = bottom_idx - top_idx
                img_w = right_idx - left_idx
                return img, bbox_sizes, bbox_labels


class RandomHorizontalFlip(BaseTransform):
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def __call__(self, img: Image, bbox_sizes: FloatTensor, bbox_labels: LongTensor):
        if random.random() < self.p:
            bbox_sizes[:, 0], bbox_sizes[:, 2] = 1.0 - bbox_sizes[:, 2], 1.0 - bbox_sizes[:, 0]
            return TF.hflip(img), bbox_sizes, bbox_labels
        return img, bbox_sizes, bbox_labels


class ColorJitter(BaseTransform):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        super().__init__()
        self.color_jitter = transforms.ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, img: Image, bbox_sizes: FloatTensor, bbox_labels: LongTensor):
        return self.color_jitter(img), bbox_sizes, bbox_labels


class UnNormalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor
