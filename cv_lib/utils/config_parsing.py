import yaml
from typing import Dict, Any


__all__ = [
    "get_cfg"
]


def get_cfg(cfg_filepath: str) -> Dict[str, Any]:
    with open(cfg_filepath) as fp:
        global_cfg = yaml.load(fp, Loader=yaml.SafeLoader)
        return global_cfg

