"""Per-iteration Sim3 trajectory recorder for registration methods.

A ``Tracer`` is an *optional* sink threaded through ``register()`` and
``semantic_icp()``. When ``None`` (the default) methods behave exactly as before,
so committed results stay reproducible. When provided, the method records the
current Sim3 every ``stride`` iterations; the evaluation script later decomposes
each recorded matrix against the frozen ``T_gt`` to plot how position / scale /
rotation error evolves over the iterations.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


class Tracer:
    def __init__(self, stride: int = 1) -> None:
        self.stride = max(int(stride), 1)
        self.steps: List[Tuple[int, np.ndarray]] = []  # (iteration index, 4x4 Sim3)

    def record(self, it: int, T: np.ndarray, force: bool = False) -> None:
        """Append the current Sim3 if ``it`` lands on the stride (or ``force``)."""
        if not (force or int(it) % self.stride == 0):
            return
        Tc = np.array(T, dtype=np.float64)
        if self.steps and self.steps[-1][0] == int(it):
            self.steps[-1] = (int(it), Tc)   # overwrite duplicate index (e.g. forced final)
        else:
            self.steps.append((int(it), Tc))
