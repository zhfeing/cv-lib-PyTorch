import abc
from collections import defaultdict
from typing import DefaultDict, Dict, Union

from torch import Tensor


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
        """Log a new value to the meter
        Args:
            value: Next restult to include.
        """
        pass

    @abc.abstractmethod
    def value(self):
        """Get the value of the meter in the current state."""
        pass


class AverageMeter(Meter):
    """
    Computes and stores the average and current value
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if isinstance(val, Tensor):
            val = val.item()
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def value(self):
        return self.avg


class DictAverageMeter(Meter):
    """
    Computes and stores the average and current value (Dict[str, Union[float, Tensor]])
    """
    def __init__(self):
        self.average_meters = defaultdict(AverageMeter)
        self.reset()

    def reset(self):
        self.average_meters.clear()

    def update(self, val: Dict[str, Union[float, Tensor]], n=1):
        for k, v in val.items():
            self.average_meters[k].update(v, n)

    def value(self):
        avg_dict = dict()
        for k, meter in self.average_meters.items():
            avg_dict[k] = meter.value()
        return avg_dict
