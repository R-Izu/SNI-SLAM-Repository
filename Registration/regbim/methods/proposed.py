"""Proposed method: physical-constraint rotation (6->3 DOF) + multi-class semantic ICP.

1. Estimate the relative rotation from labels alone (floor/ceiling -> gravity,
   wall -> Manhattan yaw). This fixes 3 of the 6 rotational/translational DOF.
2. For each of the 4 Manhattan yaw candidates, initialise translation by aligning
   structural centroids, then solve translation + scale with class-constrained
   semantic ICP (rotation held fixed -- the novelty's whole point).
3. Keep the candidate with the best semantic inlier ratio.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from .. import preprocess, rotation
from ..labels import NAME_TO_ID, LabeledCloud
from ..metrics import apply_sim3, class_inlier_ratio, make_sim3
from ..semantic_icp import semantic_icp
from .base import BaseRegistration
from . import register_method


def _structural_centroid(cloud: LabeledCloud, cfg: Dict) -> np.ndarray:
    ids = [NAME_TO_ID[n] for n in cfg["classes"]["structural"]]
    mask = np.isin(cloud.labels, ids)
    pts = cloud.points[mask] if mask.any() else cloud.points
    return pts.mean(axis=0)


@register_method("proposed")
class Proposed(BaseRegistration):
    name = "proposed"

    def register(self, src: LabeledCloud, dst: LabeledCloud, cfg: Dict) -> np.ndarray:
        src_p = preprocess.prepare(src, cfg)
        dst_p = preprocess.prepare(dst, cfg)

        c_src = _structural_centroid(src_p, cfg)
        c_dst = _structural_centroid(dst_p, cfg)
        thresh = float(cfg["semantic_icp"]["max_corr_dist"])

        candidates: List[np.ndarray] = rotation.relative_rotation_candidates(
            src_p, dst_p, cfg)
        best_T = None
        best_score = -np.inf
        for R in candidates:
            t0 = c_dst - R @ c_src                      # centroid-aligned init
            init_T = make_sim3(R, t0, 1.0)
            T = semantic_icp(src_p, dst_p, init_T, cfg, rotation_fixed=True)
            score = class_inlier_ratio(src_p, dst_p, T, thresh)
            if score > best_score:
                best_score = score
                best_T = T
        return best_T
