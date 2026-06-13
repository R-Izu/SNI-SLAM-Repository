"""Common interface every registration method implements.

A method takes two source-agnostic ``LabeledCloud`` objects and returns a 4x4
Sim3 mapping source -> reference. Because the contract is just clouds in / Sim3
out, new methods (and the future IFC reference) drop in without touching the rest.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from ..labels import LabeledCloud


class BaseRegistration:
    name: str = "base"

    def register(self, src: LabeledCloud, dst: LabeledCloud, cfg: Dict) -> np.ndarray:
        """Return a 4x4 Sim3 matrix aligning ``src`` to ``dst``."""
        raise NotImplementedError
