from typing import Any, Dict, List

from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer


class ConstantLR(_LRScheduler):
    def __init__(self, optimizer, last_epoch=-1):
        super(ConstantLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr for base_lr in self.base_lrs]


class PolynomialLR(_LRScheduler):
    def __init__(self, optimizer, max_iter, decay_iter=1, gamma=0.9, last_epoch=-1):
        self.decay_iter = decay_iter
        self.max_iter = max_iter
        self.gamma = gamma
        super(PolynomialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch % self.decay_iter or self.last_epoch % self.max_iter == 0:
            return [base_lr for base_lr in self.base_lrs]
        else:
            factor = (1 - self.last_epoch / float(self.max_iter)) ** self.gamma
            return [base_lr * factor for base_lr in self.base_lrs]


class WarmUpLR:
    def __init__(
        self,
        optimizer: Optimizer,
        scheduler: _LRScheduler,
        mode="linear",
        warmup_iters=100,
        gamma=0.2
    ):
        self.mode = mode
        self.scheduler = scheduler
        self.warmup_iters = warmup_iters
        self.gamma = gamma
        self.last_epoch = 0

        self.last_cold_lrs = self.scheduler.get_last_lr()
        self.last_lr = self.get_lr()
        self._apply_lr(self.last_lr)
        self.last_epoch = 1

    def get_last_lr(self):
        return self.last_lr

    def get_lr(self):
        if self.last_epoch < self.warmup_iters:
            if self.mode == "linear":
                alpha = self.last_epoch / float(self.warmup_iters)
                factor = self.gamma * (1 - alpha) + alpha

            elif self.mode == "constant":
                factor = self.gamma
            else:
                raise KeyError("WarmUp type {} not implemented".format(self.mode))

            return [factor * base_lr for base_lr in self.last_cold_lrs]

        return self.last_cold_lrs

    def _recover_code_lr(self):
        self._apply_lr(self.last_cold_lrs)

    def _apply_lr(self, lrs: List[float]):
        optimizer: Optimizer = self.scheduler.optimizer
        for group, lr in zip(optimizer.param_groups, lrs):
            group["lr"] = lr

    def step(self):
        self._recover_code_lr()
        self.scheduler.step()
        self.last_cold_lrs = self.scheduler.get_last_lr()
        self.last_lr = self.get_lr()
        self._apply_lr(self.last_lr)
        self.last_epoch += 1

    def state_dict(self) -> Dict[str, Any]:
        _state_dict = {
            "scheduler": self.scheduler.state_dict(),
            "mode": self.mode,
            "warmup_iters": self.warmup_iters,
            "gamma": self.gamma,
            "last_epoch": self.last_epoch,
            "last_lr": self.last_lr,
            "last_code_lrs": self.last_cold_lrs
        }

        return _state_dict

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.scheduler.load_state_dict(state_dict["scheduler"])
        self.mode = state_dict["mode"]
        self.warmup_iters = state_dict["warmup_iters"]
        self.gamma = state_dict["gamma"]
        self.last_epoch = state_dict["last_epoch"]
        self.last_lr = state_dict["last_lr"]
        self.last_cold_lrs = state_dict["last_code_lrs"]
