import abc
from collections import defaultdict
from typing import List, Dict, Union

import torch
from torch import Tensor

import cv_lib.distributed.utils as dist_utils


__all__ = [
    "Meter",
    "AverageMeter",
    "DictAverageMeter"
]


class Meter(abc.ABC):
    """
    Meters provide a way to keep track of important statistics in an online manner.
    This class is abstract, but provides a standard interface for all meters to follow.
    """
    @abc.abstractmethod
    def reset(self):
        """Resets the meter to default settings."""
        pass

    @abc.abstractmethod
    def update(self, value):
        """
        Log a new value to the meter
        Args:
            value: Next restult to include.
        """
        pass

    @abc.abstractmethod
    def value(self):
        """Get the value of the meter in the current state."""
        pass

    def sync(self):
        """Sync between multi-gpu"""
        pass

    def accumulate(self):
        """Accumulate for sync or value"""
        pass


class AverageMeter(Meter):
    """
    Computes and stores the average and current value
    """
    def __init__(self):
        self.values: List[Tensor] = list()
        self.value_accumulated: Tensor = None

    def reset(self):
        self.values.clear()
        self.value_accumulated = None

    def update(self, val: Tensor):
        self.values.append(val.to("cpu"))

    def sync(self):
        assert self.value_accumulated is not None, "`self.sync` must be called after `self.accumulate`"
        self.value_accumulated = dist_utils.reduce_tensor(self.value_accumulated, average=True)

    def accumulate(self):
        self.value_accumulated = torch.stack(self.values)

    def value(self):
        assert self.value_accumulated is not None, "`self.value` must be called after `self.accumulate`"
        return self.value_accumulated.mean(dim=0)


class DictAverageMeter(Meter):
    """
    Computes and stores the average and current value (Dict[str, Union[float, Tensor]])
    """
    def __init__(self):
        self.average_meters = defaultdict(AverageMeter)
        self.reset()

    def reset(self):
        self.average_meters.clear()

    def update(self, val: Dict[str, Union[float, Tensor]]):
        for k, v in val.items():
            self.average_meters[k].update(v)

    def sync(self):
        for v in self.average_meters.values():
            v.sync()

    def accumulate(self):
        for v in self.average_meters.values():
            v.accumulate()

    def value(self):
        avg_dict = dict()
        for k, meter in self.average_meters.items():
            avg_dict[k] = meter.value()
        return avg_dict
