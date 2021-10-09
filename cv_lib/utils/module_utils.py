import collections
from typing import List, OrderedDict, Any

from torch import nn, Tensor
from torch.utils.hooks import RemovableHandle


__all__ = [
    "MidExtractor",
    "Identity"
]


class MidExtractor:
    def __init__(
        self,
        model: nn.Module,
        extract_names: List[str],
        require_output: bool = True,
        register_now: bool = True
    ):
        self.model = model
        self.extract_names = extract_names
        self.require_output = require_output

        self.features: OrderedDict[str, Any] = collections.OrderedDict()
        self.hooks: OrderedDict[str, RemovableHandle] = collections.OrderedDict()

        for name in self.extract_names:
            self.features[name] = None
            self.hooks[name] = None

        if register_now:
            self.register_forward_hooks()

    def clear(self):
        self._remove_hooks()
        for name in self.extract_names:
            self.features[name] = None
            self.hooks[name] = None

    def register_forward_hooks(self):
        raw_model = self.model
        if isinstance(raw_model, nn.parallel.DistributedDataParallel):
            raw_model = raw_model.module
        for name, module in raw_model.named_modules():
            if name in self.extract_names:
                setattr(module, "name", name)

                def forward_hook(module: nn.Module, input, output):
                    name = getattr(module, "name")
                    feat = output if self.require_output else input
                    self.features[name] = feat

                handle = module.register_forward_hook(forward_hook)
                self.hooks[name] = handle

    def _remove_hooks(self):
        for hook in self.hooks.values():
            hook.remove()


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor):
        return x
