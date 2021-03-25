from typing import Callable, Dict, Tuple, List, Any, Optional, Union, OrderedDict
import collections

from PIL.Image import Image

from torch import Tensor
from torch.utils.data import Dataset
from torchvision.datasets.folder import pil_loader
import torchvision.transforms.functional as TF
import torchvision.ops.boxes as box_ops


class DetectionDataset(Dataset):
    """
    Base Detection Dataset

    Inhered Class Requirements:
        `img_ids`: list of img_id, the `img_id` is the unique id for each image (could be str or int)
        `images`: dict e.g. {img_id: img_filepath}
        `label_map`: OrderedDict[str, int], map a str label to its id != 0, no background
        `label_info`: OrderedDict[int, str], map a label id to its str, 0 for background
        `dataset_mean`: List[float]
        `dataset_std`: List[float]
    """
    def __init__(
            self,
            resize: Optional[Tuple[int]] = (300, 300),
            augmentations: Callable[[Image, Dict[str, Any]], Tuple[Image, Dict[str, Any]]] = None,
    ):
        """
        resize: (h, w)
        """
        self.resize = tuple(resize) if isinstance(resize, list) else resize
        self.augmentations = augmentations

        self.images: List[str] = list()
        self.label_map: OrderedDict[str, int] = collections.OrderedDict()
        self.label_info: OrderedDict[int, str] = collections.OrderedDict({0: "background"})
        self.dataset_mean: List[float] = None
        self.dataset_std: List[float] = None

    @property
    def n_classes(self) -> int:
        return len(self.label_info)

    def __len__(self) -> int:
        return len(self.images)

    def get_annotation(self, index: int) -> Dict[str, Any]:
        raise NotImplementedError

    def get_img_id(self, index: int) -> Union[str, int]:
        return index

    def __getitem__(self, index: int) -> Tuple[Tensor, Dict[str, Any]]:
        """
        Return image and target where target is a dictionary e.g.
            target: {
                image_id: str or int,
                orig_size: original image size (h, w)
                size: image size after transformation (h, w)
                boxes: relative bounding box for each object in the image (cx, cy, w, h)
                    normalized to [0, 1]
                labels: label for each bounding box
                *OTHER_INFO*: other information
            }

        Warning: after transformation, the number of bounding box of one image could be ZERO
        """
        img = pil_loader(self.images[index])
        img_w, img_h = img.size

        annotation = self.get_annotation(index)

        target: Dict[str, Any] = {
            "image_id": self.get_img_id(index),
            "orig_size": (img_h, img_w),
            "size": (img_h, img_w)
        }
        target.update(annotation)

        bbox_labels = target["labels"]
        bboxes: Tensor = target["boxes"]
        assert bboxes.shape[1] == 4 and bboxes.ndim == 2, "bound box must have shape: [n, 4]"
        # convert (xyxy)
        bboxes = box_ops.box_convert(bboxes, "xywh", "cxcywh")
        # normalize
        bboxes[:, (0, 2)] /= img_w
        bboxes[:, (1, 3)] /= img_h
        # bbox must not larger than image
        bboxes.clamp_(0, 1)

        target["boxes"] = bboxes
        target["labels"] = bbox_labels

        if self.augmentations is not None:
            img, target = self.augmentations(img, target)

        if self.resize is not None:
            img = TF.resize(img, self.resize)
            target["size"] = self.resize
        img = TF.to_tensor(img)
        img = TF.normalize(img, self.dataset_mean, self.dataset_std, inplace=True)
        return img, target

