from typing import Tuple

from torch.utils.data import Sampler, Dataset, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler


def get_sampler(
    distributed: bool,
    train_dataset: Dataset,
    val_dataset: Dataset
) -> Tuple[Sampler, Sampler]:
    if distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
    else:
        train_sampler = RandomSampler(train_dataset)
    val_sampler = SequentialSampler(val_dataset)
    return train_sampler, val_sampler

