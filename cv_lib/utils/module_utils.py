import collections
import logging
from typing import List, OrderedDict, Any

from torch import nn
from torch.utils.hooks import RemovableHandle

from cv_lib.utils import to_json_str


__all__ = [
    "MidExtractor"
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

        self.logger = logging.getLogger("mid_extractor")
        self.logger.info("Extract names:\n%s", to_json_str(self.extract_names))

        if register_now:
            self.register_forward_hooks()

    def clear(self):
        self._remove_hooks()
        for name in self.extract_names:
            self.features[name] = None
            self.hooks[name] = None

    def register_forward_hooks(self):
        for name, module in self.model.named_modules():
            if name in self.extract_names:
                setattr(module, "name", name)

                def forward_hook(module: nn.Module, input, output):
                    name = getattr(module, "name")
                    feat = output if self.require_output else input
                    self.features[name] = feat

                handle = module.register_forward_hook(forward_hook)
                self.hooks[name] = handle

    def _remove_hooks(self):
        for name, hook in self.hooks.items():
            hook.remove()
            self.logger.info("Removed hook: %s", name)

