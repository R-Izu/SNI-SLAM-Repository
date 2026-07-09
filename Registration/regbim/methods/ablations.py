"""Ablation variants of the proposed method (attribution diagnostics, not
performance improvements). Each variant disables exactly one contribution and
delegates to :class:`Proposed` with the corresponding ``cfg["ablation"]`` flag;
all method parameters stay identical to the parent.

proposed_no_semantic  -- correspondences (and yaw scoring) ignore class labels:
                         every matchable class collapses into one bucket.
                         Tests C2 (multi-class matching reduces ambiguity).
proposed_no_gravity   -- the physical-constraint stage (gravity canonicalisation,
                         Manhattan yaw candidates, plane-extent seeding) is
                         dropped; centroid init + free-rotation semantic Sim3
                         ICP. Tests the rotation-initialisation contribution.
proposed_fixed_scale  -- rigid mode: scale estimation disabled and the GT scale
                         pre-applied by the benchmark (``gt_scale`` attribute;
                         same fairness handicap the rigid baselines receive).
                         Tests C3 (joint Sim3 estimation does not hurt).
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from ..labels import LabeledCloud
from ..trace import Tracer
from .proposed import Proposed
from . import register_method


def _with_ablation(cfg: Dict, **flags) -> Dict:
    merged = {**(cfg.get("ablation") or {}), **flags}
    return {**cfg, "ablation": merged}


@register_method("proposed_no_semantic")
class ProposedNoSemantic(Proposed):
    name = "proposed_no_semantic"

    def register(self, src: LabeledCloud, dst: LabeledCloud, cfg: Dict,
                 tracer: Optional[Tracer] = None) -> np.ndarray:
        return super().register(src, dst, _with_ablation(cfg, single_class=True),
                                tracer)


@register_method("proposed_no_gravity")
class ProposedNoGravity(Proposed):
    name = "proposed_no_gravity"

    def register(self, src: LabeledCloud, dst: LabeledCloud, cfg: Dict,
                 tracer: Optional[Tracer] = None) -> np.ndarray:
        return super().register(src, dst, _with_ablation(cfg, no_gravity=True),
                                tracer)


@register_method("proposed_fixed_scale")
class ProposedFixedScale(Proposed):
    name = "proposed_fixed_scale"
    gt_scale = True   # benchmark strips the perturbation's scale component

    def register(self, src: LabeledCloud, dst: LabeledCloud, cfg: Dict,
                 tracer: Optional[Tracer] = None) -> np.ndarray:
        cfg = _with_ablation(cfg, fixed_scale=True)
        cfg = {**cfg, "semantic_icp": {**cfg["semantic_icp"], "with_scaling": False}}
        return super().register(src, dst, cfg, tracer)
