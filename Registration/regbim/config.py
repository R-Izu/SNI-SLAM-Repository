"""Config loading and persistence of the frozen reference transform T_gt."""

from __future__ import annotations

import json
import os
from typing import Dict, Optional

import numpy as np
import yaml


def load_config(path: str) -> Dict:
    with open(path) as f:
        return yaml.safe_load(f)


def save_t_gt(cfg: Dict, T: np.ndarray) -> str:
    """Persist the SLAM->reference Sim3 to the sidecar path from config."""
    path = cfg["eval"]["t_gt_path"]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"T_gt": np.asarray(T).tolist()}, f, indent=2)
    return path


def load_t_gt(cfg: Dict) -> Optional[np.ndarray]:
    """Load the frozen T_gt, or None if establish_gt.py has not run yet."""
    path = cfg["eval"]["t_gt_path"]
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return np.array(json.load(f)["T_gt"], dtype=np.float64)
