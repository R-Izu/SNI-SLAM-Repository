"""Physical-constraint rotation estimation (novelty 1).

Semantic labels make the surface assignment certain: floor/ceiling points define
the horizontal planes (hence gravity), and wall points define the vertical planes
(hence the Manhattan yaw). This collapses the 6-DOF pose problem to 3-DOF
(translation) for the downstream semantic ICP. Unlike Hübner (2021), the plane
membership is *given by the label*, not inferred from geometry alone.

The estimator brings a cloud into a canonical frame (gravity -> +Z, dominant wall
normal -> +X). The rotation aligning source to reference is then
``R_ref_canonical.T @ R_src_canonical`` (up to the 4-fold Manhattan yaw symmetry,
which the proposed pipeline disambiguates with a semantic score).
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .labels import LabeledCloud


def _normalize(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)


def _robust_axis(normals: np.ndarray, thresh_deg: float, iters: int,
                 rng: np.random.Generator) -> np.ndarray:
    """RANSAC the dominant (sign-agnostic) axis of a set of near-parallel normals."""
    n = _normalize(np.asarray(normals, dtype=np.float64))
    if len(n) == 0:
        raise ValueError("no normals for axis estimation")
    cos_t = np.cos(np.radians(thresh_deg))
    best_mask = None
    best_count = -1
    iters = int(min(iters, max(len(n), 1)))
    idx = rng.integers(0, len(n), size=iters)
    for i in idx:
        inl = np.abs(n @ n[i]) > cos_t
        c = int(inl.sum())
        if c > best_count:
            best_count = c
            best_mask = inl
    # Refine with the principal eigenvector of the inlier outer-product sum
    # (double cover handles the ± sign ambiguity of plane normals).
    M = n[best_mask].T @ n[best_mask]
    w, v = np.linalg.eigh(M)
    return _normalize(v[:, -1])


def estimate_gravity_axis(cloud: LabeledCloud, cfg: Tuple) -> np.ndarray:
    """Estimate the upward unit vector from floor + ceiling normals.

    Returns the axis oriented so that ceiling sits above floor (true 'up').
    """
    rcfg = cfg["rotation"]
    rng = np.random.default_rng(0)
    floor = cloud.subset(cloud.class_mask("floor"))
    ceiling = cloud.subset(cloud.class_mask("ceiling"))
    normals = []
    if len(floor) and floor.normals is not None:
        normals.append(floor.normals)
    if len(ceiling) and ceiling.normals is not None:
        normals.append(ceiling.normals)
    if not normals:
        raise ValueError("need floor/ceiling normals to estimate gravity")
    axis = _robust_axis(np.concatenate(normals, axis=0),
                        rcfg["ransac_normal_thresh_deg"], rcfg["ransac_iters"], rng)
    # Orient sign: ceiling mean height should exceed floor mean height.
    if len(ceiling) and len(floor):
        if (ceiling.points @ axis).mean() < (floor.points @ axis).mean():
            axis = -axis
    elif len(floor):
        # Floor normals point into the room (up); align axis with their mean.
        if (floor.normals.mean(axis=0) @ axis) < 0:
            axis = -axis
    return axis


def _horizontal_basis(up: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ref = np.array([1.0, 0.0, 0.0]) if abs(up[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = _normalize(ref - (ref @ up) * up)
    e2 = np.cross(up, e1)
    return e1, e2


def estimate_yaw_direction(cloud: LabeledCloud, up: np.ndarray, cfg: Tuple) -> np.ndarray:
    """Dominant horizontal wall-normal direction (Manhattan), as a unit vector."""
    rcfg = cfg["rotation"]
    wall = cloud.subset(cloud.class_mask("wall"))
    if len(wall) == 0 or wall.normals is None:
        raise ValueError("need wall normals to estimate yaw")
    e1, e2 = _horizontal_basis(up)
    nh = wall.normals - (wall.normals @ up)[:, None] * up      # project to horizontal
    mag = np.linalg.norm(nh, axis=1)
    nh = nh[mag > 0.3]                                          # drop near-vertical
    if len(nh) == 0:
        raise ValueError("no horizontal wall normals found")
    nh = _normalize(nh)
    ang = np.arctan2(nh @ e2, nh @ e1)                         # in [-pi, pi]
    ang_mod = np.mod(ang, np.pi / 2.0)                         # Manhattan: collapse to [0, pi/2)
    bins = int(rcfg["yaw_bins"])
    hist, edges = np.histogram(ang_mod, bins=bins, range=(0.0, np.pi / 2.0))
    phi = edges[int(np.argmax(hist))] + (np.pi / 2.0) / bins / 2.0
    return _normalize(np.cos(phi) * e1 + np.sin(phi) * e2)


def canonical_rotation(cloud: LabeledCloud, cfg: Tuple) -> np.ndarray:
    """Rotation mapping the cloud into its canonical frame (up->+Z, wall->+X)."""
    up = estimate_gravity_axis(cloud, cfg)
    xdir = estimate_yaw_direction(cloud, up, cfg)
    z = up
    x = _normalize(xdir - (xdir @ z) * z)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=0)                        # rows are the canonical axes


def _rot_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def relative_rotation_candidates(src: LabeledCloud, ref: LabeledCloud,
                                 cfg: Tuple) -> List[np.ndarray]:
    """The 4 Manhattan-symmetric rotations aligning src to ref (yaw ambiguity)."""
    Rs = canonical_rotation(src, cfg)
    Rr = canonical_rotation(ref, cfg)
    return [Rr.T @ _rot_z(k * np.pi / 2.0) @ Rs for k in range(4)]


def estimate_rotation(src: LabeledCloud, ref: LabeledCloud, cfg: Tuple) -> np.ndarray:
    """Primary relative rotation (k=0 candidate)."""
    return relative_rotation_candidates(src, ref, cfg)[0]
