"""Downsampling, normal estimation, and per-class splitting for ``LabeledCloud``."""

from __future__ import annotations

from typing import Dict

import numpy as np
import open3d as o3d

from .labels import LabeledCloud


def estimate_normals(cloud: LabeledCloud, radius: float, max_nn: int) -> LabeledCloud:
    """Estimate normals if missing (in place semantics, returns same cloud)."""
    if cloud.normals is not None:
        return cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud.points)
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    cloud.normals = np.asarray(pcd.normals)
    return cloud


def voxel_downsample(cloud: LabeledCloud, voxel_size: float) -> LabeledCloud:
    """Voxel-downsample per class so labels (and normals) are preserved exactly."""
    pts_out, lbl_out, nrm_out = [], [], []
    has_normals = cloud.normals is not None
    for c in cloud.present_classes():
        mask = cloud.labels == c
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cloud.points[mask])
        if has_normals:
            pcd.normals = o3d.utility.Vector3dVector(cloud.normals[mask])
        ds = pcd.voxel_down_sample(voxel_size)
        p = np.asarray(ds.points)
        pts_out.append(p)
        lbl_out.append(np.full(len(p), c, dtype=np.int64))
        if has_normals:
            nrm_out.append(np.asarray(ds.normals))
    return LabeledCloud(
        points=np.concatenate(pts_out, axis=0),
        labels=np.concatenate(lbl_out, axis=0),
        normals=np.concatenate(nrm_out, axis=0) if has_normals else None,
        meta=dict(cloud.meta),
    )


def split_by_label(cloud: LabeledCloud) -> Dict[int, LabeledCloud]:
    """Split into per-class sub-clouds keyed by class id."""
    return {c: cloud.subset(cloud.labels == c) for c in cloud.present_classes()}


def prepare(cloud: LabeledCloud, cfg: Dict) -> LabeledCloud:
    """Standard pipeline prep: ensure normals, then voxel-downsample."""
    pp = cfg["preprocess"]
    cloud = estimate_normals(cloud, pp["normal_radius"], pp["normal_max_nn"])
    return voxel_downsample(cloud, pp["voxel_size"])
