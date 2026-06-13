"""Multi-class semantic ICP (novelty 2).

Correspondences are formed *only within the same class* (wall<->wall,
floor<->floor, ...): a point never matches across classes. This label constraint
is what lets the registration survive repetitive indoor structure and the
asymmetry between the SLAM cloud (which has furniture/background) and the
BIM-surrogate reference (structural classes only) -- background points simply
have no same-class partner and drop out.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from scipy.spatial import cKDTree

from .labels import NAME_TO_ID, LabeledCloud
from .metrics import apply_sim3, decompose_sim3, make_sim3
from .scale import scale_translation, umeyama


def _tukey_weights(dist: np.ndarray, c: float) -> np.ndarray:
    """Tukey biweight on residual distance; zero beyond scale ``c``."""
    r = np.clip(dist / max(c, 1e-9), 0.0, 1.0)
    return (1.0 - r ** 2) ** 2


def semantic_icp(
    src: LabeledCloud,
    dst: LabeledCloud,
    init_T: np.ndarray,
    cfg: Dict,
    rotation_fixed: Optional[bool] = None,
) -> np.ndarray:
    """Iterative class-constrained Sim3 refinement. Returns a 4x4 Sim3 matrix."""
    icfg = cfg["semantic_icp"]
    if rotation_fixed is None:
        rotation_fixed = bool(icfg.get("rotation_fixed", False))
    max_corr = float(icfg["max_corr_dist"])
    tukey_c = float(icfg["tukey_c"])
    max_iter = int(icfg["max_iter"])
    conv = float(icfg["convergence_delta"])
    with_scaling = bool(icfg["with_scaling"])

    match_ids = [NAME_TO_ID[n] for n in cfg["classes"]["match_classes"]]
    common = sorted(set(src.present_classes()) & set(dst.present_classes())
                    & set(match_ids))

    # Per-class KD-trees over the destination (built once).
    dst_pts = {c: dst.points[dst.labels == c] for c in common}
    trees = {c: cKDTree(dst_pts[c]) for c in common if len(dst_pts[c]) > 0}
    src_pts = {c: src.points[src.labels == c] for c in common}

    T = np.array(init_T, dtype=np.float64)
    R_fixed, _, _ = decompose_sim3(T)

    for _ in range(max_iter):
        moved = apply_sim3(T, src.points)
        s_list, d_list, w_list = [], [], []
        for c in common:
            if c not in trees or len(src_pts[c]) == 0:
                continue
            sm = src.labels == c
            dist, idx = trees[c].query(moved[sm], k=1, workers=-1)
            keep = dist < max_corr
            if keep.sum() < 1:
                continue
            s_list.append(src.points[sm][keep])
            d_list.append(dst_pts[c][idx[keep]])
            w_list.append(_tukey_weights(dist[keep], tukey_c))
        if not s_list:
            break
        src_corr = np.concatenate(s_list, axis=0)
        dst_corr = np.concatenate(d_list, axis=0)
        w = np.concatenate(w_list, axis=0)
        if len(src_corr) < 3 or w.sum() < 1e-9:
            break

        if rotation_fixed:
            src_rot = src_corr @ R_fixed.T
            t, s = scale_translation(src_rot, dst_corr, w)
            if not with_scaling:
                s = 1.0
            T_new = make_sim3(R_fixed, t, s)
        else:
            R, t, s = umeyama(src_corr, dst_corr, w, with_scaling=with_scaling)
            T_new = make_sim3(R, t, s)

        delta = float(np.linalg.norm(T_new - T))
        T = T_new
        if delta < conv:
            break
    return T
