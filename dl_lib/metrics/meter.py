import abc

from torch import Tensor


__all__ = [
    "Meter",
    "AverageMeter",
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
    """Computes and stores the average and current value"""

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
