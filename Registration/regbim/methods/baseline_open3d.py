"""Baseline: pure-geometry Open3D global registration (no semantics).

FPFH features -> RANSAC global alignment -> ICP refinement with scaling. Uses no
labels at all, so it is the honest comparison point for "what do the physical /
label constraints buy us?". Operates on all points (geometry only).
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import open3d as o3d

from ..labels import LabeledCloud
from .base import BaseRegistration
from . import register_method


def _prep(points: np.ndarray, voxel: float, fpfh_radius: float, fpfh_max_nn: int):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd = pcd.voxel_down_sample(voxel)
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2.0, max_nn=30))
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=fpfh_radius, max_nn=fpfh_max_nn))
    return pcd, fpfh


@register_method("baseline_open3d")
class BaselineOpen3D(BaseRegistration):
    name = "baseline_open3d"

    def register(self, src: LabeledCloud, dst: LabeledCloud, cfg: Dict) -> np.ndarray:
        b = cfg["baseline_open3d"]
        src_pcd, src_fpfh = _prep(src.points, b["voxel_size"], b["fpfh_radius"], b["fpfh_max_nn"])
        dst_pcd, dst_fpfh = _prep(dst.points, b["voxel_size"], b["fpfh_radius"], b["fpfh_max_nn"])

        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            src_pcd, dst_pcd, src_fpfh, dst_fpfh, True, b["ransac_dist"],
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            b["ransac_n"],
            [o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(b["ransac_dist"])],
            o3d.pipelines.registration.RANSACConvergenceCriteria(
                int(b["ransac_iters"]), b["ransac_confidence"]),
        )
        # ICP refinement WITH scaling -> full Sim3.
        icp = o3d.pipelines.registration.registration_icp(
            src_pcd, dst_pcd, b["icp_dist"], result.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(True),
        )
        return np.array(icp.transformation, dtype=np.float64)
