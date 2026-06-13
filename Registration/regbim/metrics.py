"""Sim3 utilities, error metrics, chamfer distance, and perturbation generation."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import open3d as o3d

from .labels import CLASS_NAMES, LabeledCloud


# --------------------------------------------------------------------------- #
# Sim3 representation: 4x4 homogeneous, top-left = s*R, applies x -> s R x + t.
# --------------------------------------------------------------------------- #
def make_sim3(R: np.ndarray, t: np.ndarray, s: float) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = s * np.asarray(R)
    T[:3, 3] = np.asarray(t)
    return T


def apply_sim3(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    return points @ T[:3, :3].T + T[:3, 3]


def decompose_sim3(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Return (R, t, s) from a Sim3 matrix (assumes isotropic scale)."""
    M = T[:3, :3]
    s = float(np.cbrt(max(np.linalg.det(M), 1e-18)))
    R = M / s
    # Re-orthonormalise to guard against numerical drift.
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    return R, T[:3, 3].copy(), s


def rotation_error_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    R = R_a @ R_b.T
    cos = (np.trace(R) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def translation_error(t_a: np.ndarray, t_b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(t_a) - np.asarray(t_b)))


def scale_error_ratio(s_a: float, s_b: float) -> float:
    """Symmetric relative scale error |s_a/s_b - 1| (0 == identical)."""
    return float(abs(s_a / s_b - 1.0))


def sim3_errors(T_est: np.ndarray, T_gt: np.ndarray) -> Dict[str, float]:
    Re, te, se = decompose_sim3(T_est)
    Rg, tg, sg = decompose_sim3(T_gt)
    return {
        "rot_deg": rotation_error_deg(Re, Rg),
        "trans": translation_error(te, tg),
        "scale_ratio": scale_error_ratio(se, sg),
    }


def angle_between(u: np.ndarray, v: np.ndarray) -> float:
    """Angle (deg) between two vectors, sign-agnostic (treats ±v as equal)."""
    u = u / (np.linalg.norm(u) + 1e-12)
    v = v / (np.linalg.norm(v) + 1e-12)
    return float(np.degrees(np.arccos(np.clip(abs(u @ v), -1.0, 1.0))))


# --------------------------------------------------------------------------- #
# Alignment-quality metrics (no GT transform needed).
# --------------------------------------------------------------------------- #
def chamfer_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Symmetric mean nearest-neighbour distance between point sets a and b."""
    pa = o3d.geometry.PointCloud(); pa.points = o3d.utility.Vector3dVector(np.asarray(a))
    pb = o3d.geometry.PointCloud(); pb.points = o3d.utility.Vector3dVector(np.asarray(b))
    d_ab = np.asarray(pa.compute_point_cloud_distance(pb))
    d_ba = np.asarray(pb.compute_point_cloud_distance(pa))
    return float(d_ab.mean() + d_ba.mean()) / 2.0


def class_inlier_ratio(src: LabeledCloud, dst: LabeledCloud, T: np.ndarray,
                       thresh: float) -> float:
    """Fraction of transformed src points with a same-class dst point within thresh.

    Measures semantic alignment quality directly; needs no GT transform.
    """
    from scipy.spatial import cKDTree

    moved = apply_sim3(T, src.points)
    total = 0
    inliers = 0
    common = set(src.present_classes()) & set(dst.present_classes())
    for c in common:
        s_mask = src.labels == c
        d_mask = dst.labels == c
        if s_mask.sum() == 0 or d_mask.sum() == 0:
            continue
        tree = cKDTree(dst.points[d_mask])
        dist, _ = tree.query(moved[s_mask], k=1, workers=-1)
        total += len(dist)
        inliers += int((dist < thresh).sum())
    return float(inliers / max(total, 1))


# --------------------------------------------------------------------------- #
# Robustness perturbation
# --------------------------------------------------------------------------- #
def random_sim3(rng: np.random.Generator, ranges: Dict) -> np.ndarray:
    """Sample a known Sim3 within configured ranges (for recovery experiments)."""
    axis = rng.normal(size=3)
    axis /= (np.linalg.norm(axis) + 1e-12)
    ang = np.radians(rng.uniform(*ranges["rot_deg"]))
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)  # Rodrigues
    t = rng.uniform(ranges["trans"][0], ranges["trans"][1], size=3)
    s = float(np.exp(rng.uniform(*ranges["log_scale"])))
    return make_sim3(R, t, s)


def transform_cloud(cloud: LabeledCloud, T: np.ndarray) -> LabeledCloud:
    """Apply a Sim3 to a cloud: points via s R x + t, normals via R only."""
    R, _, _ = decompose_sim3(T)
    normals = None if cloud.normals is None else cloud.normals @ R.T
    return LabeledCloud(points=apply_sim3(T, cloud.points), labels=cloud.labels.copy(),
                        normals=normals, meta=dict(cloud.meta))


def invert_sim3(T: np.ndarray) -> np.ndarray:
    """Inverse of a Sim3 matrix."""
    R, t, s = decompose_sim3(T)
    Ri = R.T
    si = 1.0 / s
    ti = -si * (Ri @ t)
    return make_sim3(Ri, ti, si)


def class_count_table(cloud: LabeledCloud) -> Dict[str, int]:
    return {CLASS_NAMES[c]: int((cloud.labels == c).sum())
            for c in cloud.present_classes()}
