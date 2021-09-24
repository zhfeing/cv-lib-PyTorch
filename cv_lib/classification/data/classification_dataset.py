from typing import Callable, Dict, Tuple, List, Any, Optional, Union, OrderedDict
import collections

from PIL.Image import Image

from torch import Tensor
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


class ClassificationDataset(Dataset):
    """
    Base Detection Dataset

    Inhered Class Requirements:
        `img_ids`: list of img_id, the `img_id` is the unique id for each image must be string
        `label_map`: OrderedDict[str, int], map a str label to its id
        `label_info`: OrderedDict[int, str], map a label id to its str
        `dataset_mean`: List[float]
        `dataset_std`: List[float]
    """
    def __init__(
            self,
            resize: Optional[Tuple[int]] = (224, 224),
            augmentations: Callable[[Image, Dict[str, Any]], Tuple[Image, Dict[str, Any]]] = None,
    ):
        """
        resize: (h, w)
        """
        self.resize = tuple(resize) if isinstance(resize, list) else resize
        self.augmentations = augmentations

        self.label_map: OrderedDict[str, int] = collections.OrderedDict()
        self.label_info: OrderedDict[int, str] = collections.OrderedDict()
        self.dataset_mean: List[float] = None
        self.dataset_std: List[float] = None

    @property
    def n_classes(self) -> int:
        return len(self.label_info)

    def get_image(self, index: int) -> Image:
        raise NotImplementedError

    def get_annotation(self, index: int) -> Dict[str, Any]:
        raise NotImplementedError

    def get_img_id(self, index: int) -> Union[int, str]:
        return index

    def __getitem__(self, index: int) -> Tuple[Tensor, Dict[str, Any]]:
        """
        Return image and target where target is a dictionary e.g.
            target: {
                image_id: str or int
                label: LongTensor
                *OTHER_INFO*: other information
            }

        Warning: after transformation, the number of bounding box of one image could be ZERO
        """
        img = self.get_image(index)
        annotation = self.get_annotation(index)

        target: Dict[str, Any] = {
            "image_id": self.get_img_id(index),
        }
        target.update(annotation)

        if self.augmentations is not None:
            img, target = self.augmentations(img, target)

        if self.resize is not None:
            img = TF.resize(img, self.resize)
        img = TF.to_tensor(img)
        img = TF.normalize(img, self.dataset_mean, self.dataset_std, inplace=True)
        return img, target

