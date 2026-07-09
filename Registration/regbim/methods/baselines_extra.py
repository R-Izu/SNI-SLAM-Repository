"""Additional pure-geometry baselines (journal eval Phase 3).

Coarse: FGR (Open3D built-in) or FPFH+RANSAC (as in baseline_open3d).
Dense refinement: ICP point-to-point or point-to-plane (zhangcpm's two types).

All methods here are RIGID: they cannot estimate scale, so they declare
``gt_scale = True`` and the benchmark pre-applies the GT (perturbation) scale —
the fairness handicap documented in the evaluation protocol. Parameters are
read from ``cfg["baseline_extra"]`` when present, defaulting to the
``baseline_open3d`` block so existing scene configs work unchanged.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import open3d as o3d

from ..labels import LabeledCloud
from ..trace import Tracer
from .base import BaseRegistration
from .baseline_open3d import _prep
from . import register_method


def _params(cfg: Dict) -> Dict:
    p = dict(cfg["baseline_open3d"])
    p.update(cfg.get("baseline_extra") or {})
    return p


def _refine_icp(src_pcd, dst_pcd, T0: np.ndarray, dist: float, max_it: int,
                point_to_plane: bool) -> np.ndarray:
    if point_to_plane:
        est = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    else:
        est = o3d.pipelines.registration.TransformationEstimationPointToPoint(False)
    icp = o3d.pipelines.registration.registration_icp(
        src_pcd, dst_pcd, dist, T0, est,
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_it))
    return np.array(icp.transformation, dtype=np.float64)


class _FGRBase(BaseRegistration):
    """FGR coarse + rigid ICP refinement (dense type set by subclass)."""

    gt_scale = True
    point_to_plane = False

    def register(self, src: LabeledCloud, dst: LabeledCloud, cfg: Dict,
                 tracer: Optional[Tracer] = None) -> np.ndarray:
        b = _params(cfg)
        src_pcd, src_fpfh = _prep(src.points, b["voxel_size"], b["fpfh_radius"],
                                  b["fpfh_max_nn"])
        dst_pcd, dst_fpfh = _prep(dst.points, b["voxel_size"], b["fpfh_radius"],
                                  b["fpfh_max_nn"])
        opt = o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=float(b.get("fgr_dist", b["ransac_dist"])))
        result = o3d.pipelines.registration.registration_fast_based_on_feature_matching(
            src_pcd, dst_pcd, src_fpfh, dst_fpfh, opt)
        T = _refine_icp(src_pcd, dst_pcd,
                        np.array(result.transformation, dtype=np.float64),
                        float(b["icp_dist"]), int(b.get("icp_iters", 30)),
                        self.point_to_plane)
        if tracer is not None:
            tracer.record(0, T, force=True)
        return T


@register_method("baseline_fgr")
class BaselineFGR(_FGRBase):
    name = "baseline_fgr"
    point_to_plane = False


@register_method("baseline_fgr_p2l")
class BaselineFGRP2L(_FGRBase):
    name = "baseline_fgr_p2l"
    point_to_plane = True


@register_method("baseline_ransac_p2l")
class BaselineRansacP2L(BaseRegistration):
    """FPFH+RANSAC coarse (as baseline_open3d) + rigid point-to-plane ICP."""

    name = "baseline_ransac_p2l"
    gt_scale = True

    def register(self, src: LabeledCloud, dst: LabeledCloud, cfg: Dict,
                 tracer: Optional[Tracer] = None) -> np.ndarray:
        b = _params(cfg)
        src_pcd, src_fpfh = _prep(src.points, b["voxel_size"], b["fpfh_radius"],
                                  b["fpfh_max_nn"])
        dst_pcd, dst_fpfh = _prep(dst.points, b["voxel_size"], b["fpfh_radius"],
                                  b["fpfh_max_nn"])
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            src_pcd, dst_pcd, src_fpfh, dst_fpfh, True, b["ransac_dist"],
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            b["ransac_n"],
            [o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(b["ransac_dist"])],
            o3d.pipelines.registration.RANSACConvergenceCriteria(
                int(b["ransac_iters"]), b["ransac_confidence"]))
        T = _refine_icp(src_pcd, dst_pcd,
                        np.array(result.transformation, dtype=np.float64),
                        float(b["icp_dist"]), int(b.get("icp_iters", 30)),
                        point_to_plane=True)
        if tracer is not None:
            tracer.record(0, T, force=True)
        return T
