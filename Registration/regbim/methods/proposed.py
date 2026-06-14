"""Proposed method: physical-constraint rotation (6->3 DOF) + multi-class semantic
ICP, with a two-stage plane-constrained translation/scale initialisation.

1. Estimate the relative rotation from labels alone (floor/ceiling -> gravity,
   wall -> Manhattan yaw). This fixes 3 of the 6 rotational/translational DOF and
   leaves the 4-fold Manhattan yaw ambiguity.

2. Stage 1 -- lock yaw + scale from structural planes. For each yaw candidate,
   seed the translation and isotropic scale from the per-axis structural extents
   in the reference frame (floor+ceiling bound the vertical extent -> room height
   -> scale + height offset; walls bound the two horizontal extents -> room
   footprint -> xy scale + offset), then run rotation-fixed semantic ICP. The
   yaw whose run has the best semantic inlier ratio wins. The plane extents make
   each yaw candidate's init distinct, which is what disambiguates the Manhattan
   symmetry, and the extent ratios give a scale-aware start so the alignment
   survives a wrong initial scale.

3. Stage 2 -- refine the translation. The wall extents that disambiguate yaw also
   bias the translation when the source only partially covers a wall (the SLAM
   cloud vs the complete reference). So, holding the locked rotation and scale,
   re-seed the translation from the structural centroid and run one more
   rotation-fixed ICP, then keep whichever of {stage-1 plane result, centroid
   refinement} has the lower chamfer. Both share the locked (R, s) and differ
   only in the translation basin, so this chamfer tie-break is safe -- it keeps
   the centroid refinement when it tightens the fit and falls back to the plane
   result otherwise.

Correspondences throughout are class-constrained (semantic ICP, novelty 2): a
point only ever matches a same-class point, which lets the alignment survive
repetitive indoor structure and the asymmetry between the SLAM cloud and the
structural-only reference.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .. import preprocess, rotation
from ..labels import NAME_TO_ID, LabeledCloud
from ..metrics import (apply_sim3, chamfer_distance, class_inlier_ratio,
                       decompose_sim3, make_sim3)
from ..semantic_icp import semantic_icp
from ..trace import Tracer
from .base import BaseRegistration
from . import register_method

# Robust span percentiles: trims a few % of stragglers (mislabelled points,
# edge normal-estimation noise) before reading a structural extent.
_LO, _HI = 2.0, 98.0
# An axis informs scale only if its structural span is room-sized rather than
# label noise (metres).
_MIN_EXTENT = 0.5
# Guard a degenerate extent ratio from injecting a wild init scale.
_SCALE_CLIP = (0.3, 3.0)


def _axis_span(points: np.ndarray, labels: np.ndarray, class_ids: List[int],
               axis: np.ndarray) -> Optional[Tuple[float, float]]:
    """Robust (center, extent) of the given classes projected onto ``axis``."""
    mask = np.isin(labels, class_ids)
    if mask.sum() < 10:
        return None
    proj = points[mask] @ axis
    lo, hi = np.percentile(proj, [_LO, _HI])
    return 0.5 * (lo + hi), float(hi - lo)


def _struct_centroid(points: np.ndarray, labels: np.ndarray,
                     struct_ids: List[int]) -> np.ndarray:
    mask = np.isin(labels, struct_ids)
    return points[mask].mean(axis=0) if mask.any() else points.mean(axis=0)


def _chamfer(T: np.ndarray, src: LabeledCloud, dst: LabeledCloud) -> float:
    return chamfer_distance(apply_sim3(T, src.points), dst.points)


@register_method("proposed")
class Proposed(BaseRegistration):
    name = "proposed"

    def register(self, src: LabeledCloud, dst: LabeledCloud, cfg: Dict,
                 tracer: Optional[Tracer] = None) -> np.ndarray:
        src_p = preprocess.prepare(src, cfg)
        dst_p = preprocess.prepare(dst, cfg)
        thresh = float(cfg["semantic_icp"]["max_corr_dist"])

        struct_ids = [NAME_TO_ID[n] for n in cfg["classes"]["structural"]]
        floor_ceiling = [NAME_TO_ID["floor"], NAME_TO_ID["ceiling"]]
        wall = [NAME_TO_ID["wall"]]

        # Reference canonical axes (rows: wall-x, wall-y, up) in the reference's
        # own frame -- the frame each rotation candidate maps the source into.
        Rr = rotation.canonical_rotation(dst_p, cfg)
        axes = [("x", Rr[0], wall), ("y", Rr[1], wall), ("up", Rr[2], floor_ceiling)]
        dst_span = {k: _axis_span(dst_p.points, dst_p.labels, ids, e)
                    for k, e, ids in axes}
        c_dst = _struct_centroid(dst_p.points, dst_p.labels, struct_ids)

        # --- stage 1: plane-seeded search -> reliable yaw R and scale s ----------
        candidates: List[np.ndarray] = rotation.relative_rotation_candidates(
            src_p, dst_p, cfg)
        plane_T = None
        plane_trace: Optional[Tracer] = None
        best_score = -np.inf
        for R in candidates:
            src_rot = src_p.points @ R.T                   # source in reference frame
            c_src = _struct_centroid(src_rot, src_p.labels, struct_ids)
            init_T = self._plane_seed(src_rot, src_p.labels, R, axes, dst_span,
                                      c_dst, c_src)
            # Each yaw candidate gets its own tracer; only the winner's trajectory
            # is surfaced, so the reported curve is a single coherent ICP run.
            cand_tracer = Tracer(tracer.stride) if tracer is not None else None
            T = semantic_icp(src_p, dst_p, init_T, cfg, rotation_fixed=True,
                             tracer=cand_tracer)
            score = class_inlier_ratio(src_p, dst_p, T, thresh)
            if score > best_score:
                best_score = score
                plane_T = T
                plane_trace = cand_tracer

        # --- stage 2: centroid-refine translation at the locked (R, s) -----------
        R_win, _, s_win = decompose_sim3(plane_T)
        c_src_win = _struct_centroid(src_p.points @ R_win.T, src_p.labels, struct_ids)
        refine_T0 = make_sim3(R_win, c_dst - s_win * c_src_win, s_win)
        refine_trace = Tracer(tracer.stride) if tracer is not None else None
        refine_T = semantic_icp(src_p, dst_p, refine_T0, cfg, rotation_fixed=True,
                                tracer=refine_trace)

        # Pick by chamfer: safe here because both poses share the locked (R, s)
        # and differ only in translation basin (no cross-yaw/scale ambiguity).
        cand = [(_chamfer(plane_T, src_p, dst_p), plane_T, plane_trace),
                (_chamfer(refine_T, src_p, dst_p), refine_T, refine_trace)]
        _, win_T, win_trace = min(cand, key=lambda x: x[0])
        if tracer is not None and win_trace is not None:
            tracer.steps = win_trace.steps
        return win_T

    @staticmethod
    def _plane_seed(src_rot: np.ndarray, labels: np.ndarray, R: np.ndarray, axes,
                    dst_span, c_dst: np.ndarray, c_src: np.ndarray) -> np.ndarray:
        """Scale + translation from per-axis structural extents (rotation fixed)."""
        src_span = {k: _axis_span(src_rot, labels, ids, e) for k, e, ids in axes}

        # Isotropic scale = median of the per-axis extent ratios (robust to one
        # partial/occluded axis); fall back to 1.0 if no axis is informative.
        ratios = [dst_span[k][1] / src_span[k][1]
                  for k, _, _ in axes
                  if dst_span[k] is not None and src_span[k] is not None
                  and src_span[k][1] > _MIN_EXTENT and dst_span[k][1] > _MIN_EXTENT]
        s = float(np.clip(np.median(ratios), *_SCALE_CLIP)) if ratios else 1.0

        # Centroid alignment at the estimated scale, then overwrite each axis
        # component with its plane-matched offset where available (guarantees a
        # full vector even if an axis is missing).
        t = c_dst - s * c_src
        for k, e, _ in axes:
            if dst_span[k] is not None and src_span[k] is not None:
                cd, cs = dst_span[k][0], src_span[k][0]
                t = t - (t @ e) * e + (cd - s * cs) * e
        return make_sim3(R, t, s)
