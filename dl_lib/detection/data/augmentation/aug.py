"""
All bounding boxes are supposed as "cxcywh" format and normalized to [0, 1]
"""

import random
from typing import Tuple, Dict, Any, List
import abc

from PIL.Image import Image

import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from .functional import crop, pad_bottom_right, resize


__all__ = [
    "BaseTransform",
    "Compose",
    "RandomHorizontalFlip",
    "ColorJitter",
    "UnNormalize",
    "RandomCrop",
    "RandomSizeCrop",
    "CenterCrop",
    "RandomPadBottomRight",
    "RandomResize",
    "RandomSelect",
    "RandomErasing"
]


class BaseTransform(abc.ABC):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def __call__(
        self,
        img: Image,
        target: Dict[str, Any]
    ) -> Tuple[Image, Dict[str, Any]]:
        pass


class Compose(BaseTransform):
    def __init__(self, *transforms):
        super().__init__()
        self.transforms = transforms

    def __call__(
        self,
        img: Image,
        target: Dict[str, Any]
    ) -> Tuple[Image, Dict[str, Any]]:
        for tf in self.transforms:
            img, target = tf(img, target)
        return img, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class RandomHorizontalFlip(BaseTransform):
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def __call__(
        self,
        img: Image,
        target: Dict[str, Any]
    ) -> Tuple[Image, Dict[str, Any]]:
        if random.random() < self.p:
            img = TF.hflip(img)
            if "boxes" in target:
                boxes = target["boxes"]
                target["boxes"][:, 0] = 1 - boxes[:, 0]
            if "masks" in target:
                target["masks"] = TF.hflip(target["masks"])
        return img, target


class ColorJitter(BaseTransform):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        super().__init__()
        self.color_jitter = transforms.ColorJitter(brightness, contrast, saturation, hue)

    def __call__(
        self,
        img: Image,
        target: Dict[str, Any]
    ) -> Tuple[Image, Dict[str, Any]]:
        return self.color_jitter(img), target


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


class RandomCrop(BaseTransform):
    def __init__(self, size: Tuple[int, int]):
        self.size = size

    def __call__(
        self,
        img: Image,
        target: Dict[str, Any]
    ) -> Tuple[Image, Dict[str, Any]]:
        region = transforms.RandomCrop.get_params(img, self.size)
        return crop(img, target, region)


class RandomSizeCrop(BaseTransform):
    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(
        self,
        img: Image,
        target: Dict[str, Any]
    ) -> Tuple[Image, Dict[str, Any]]:
        w = random.randint(self.min_size, min(img.width, self.max_size))
        h = random.randint(self.min_size, min(img.height, self.max_size))
        region = transforms.RandomCrop.get_params(img, [h, w])
        return crop(img, target, region)


class CenterCrop(BaseTransform):
    def __init__(self, size: Tuple[int, int]):
        self.size = size

    def __call__(
        self,
        img: Image,
        target: Dict[str, Any]
    ) -> Tuple[Image, Dict[str, Any]]:
        image_width, image_height = img.size
        crop_height, crop_width = self.size
        crop_top = int(round((image_height - crop_height) / 2.))
        crop_left = int(round((image_width - crop_width) / 2.))
        return crop(img, target, (crop_top, crop_left, crop_height, crop_width))


class RandomResize(BaseTransform):
    def __init__(self, sizes: List[int], max_size=None):
        """
        Args:
            sizes: list of size to be chosen to resize the image
        """
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(
        self,
        img: Image,
        target: Dict[str, Any]
    ) -> Tuple[Image, Dict[str, Any]]:
        size = random.choice(self.sizes)
        return resize(img, target, size, self.max_size)


class RandomPadBottomRight(BaseTransform):
    def __init__(self, max_pad: int):
        self.max_pad = max_pad

    def __call__(
        self,
        img: Image,
        target: Dict[str, Any]
    ) -> Tuple[Image, Dict[str, Any]]:
        pad_x = random.randint(0, self.max_pad)
        pad_y = random.randint(0, self.max_pad)
        return pad_bottom_right(img, target, (pad_x, pad_y))


class RandomSelect(BaseTransform):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """
    def __init__(self, transforms1: BaseTransform, transforms2: BaseTransform, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(
        self,
        img: Image,
        target: Dict[str, Any]
    ) -> Tuple[Image, Dict[str, Any]]:
        if random.random() < self.p:
            return self.transforms1(img, target)
        return self.transforms2(img, target)


class RandomErasing(BaseTransform):
    def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False):
        self.eraser = transforms.RandomErasing(p, scale, ratio, value, inplace)

    def __call__(
        self,
        img: Image,
        target: Dict[str, Any]
    ) -> Tuple[Image, Dict[str, Any]]:
        return self.eraser(img), target

