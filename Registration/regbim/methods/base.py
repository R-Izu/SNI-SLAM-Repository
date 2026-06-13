"""Common interface every registration method implements.

A method takes two source-agnostic ``LabeledCloud`` objects and returns a 4x4
Sim3 mapping source -> reference. Because the contract is just clouds in / Sim3
out, new methods (and the future IFC reference) drop in without touching the rest.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from ..labels import LabeledCloud
from ..trace import Tracer


class BaseRegistration:
    name: str = "base"

    def register(self, src: LabeledCloud, dst: LabeledCloud, cfg: Dict,
                 tracer: Optional[Tracer] = None) -> np.ndarray:
        """Return a 4x4 Sim3 matrix aligning ``src`` to ``dst``.

        ``tracer`` is optional: when given, the method records its per-iteration
        Sim3 trajectory into it (for error-vs-GT analysis). When ``None`` the
        method must behave exactly as if tracing did not exist.
        """
        raise NotImplementedError
