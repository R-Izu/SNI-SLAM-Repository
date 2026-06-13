"""Synthetic box-room ``LabeledCloud`` for deterministic geometry tests."""

import numpy as np

from regbim.labels import NAME_TO_ID, LabeledCloud


def make_room(n_per_face: int = 1500, seed: int = 0,
              lx: float = 4.0, ly: float = 3.0, lz: float = 2.5) -> LabeledCloud:
    """Axis-aligned box: floor (+Z normal), ceiling (-Z normal), 4 walls (±X, ±Y)."""
    rng = np.random.default_rng(seed)
    pts, nrm, lab = [], [], []

    def add(p, normal, name):
        pts.append(p)
        nrm.append(np.tile(normal, (len(p), 1)))
        lab.append(np.full(len(p), NAME_TO_ID[name]))

    u = lambda a, b: rng.uniform(a, b, n_per_face)
    z0, z1 = 0.0, lz
    # floor / ceiling
    add(np.stack([u(0, lx), u(0, ly), np.full(n_per_face, z0)], 1), [0, 0, 1], "floor")
    add(np.stack([u(0, lx), u(0, ly), np.full(n_per_face, z1)], 1), [0, 0, -1], "ceiling")
    # walls
    add(np.stack([np.full(n_per_face, 0.0), u(0, ly), u(z0, z1)], 1), [1, 0, 0], "wall")
    add(np.stack([np.full(n_per_face, lx), u(0, ly), u(z0, z1)], 1), [-1, 0, 0], "wall")
    add(np.stack([u(0, lx), np.full(n_per_face, 0.0), u(z0, z1)], 1), [0, 1, 0], "wall")
    add(np.stack([u(0, lx), np.full(n_per_face, ly), u(z0, z1)], 1), [0, -1, 0], "wall")

    points = np.concatenate(pts, 0)
    normals = np.concatenate(nrm, 0).astype(float)
    labels = np.concatenate(lab, 0)
    # mild noise so estimators face non-perfect input
    points = points + rng.normal(0, 0.005, points.shape)
    return LabeledCloud(points=points, labels=labels, normals=normals)


def axis_angle(axis, deg: float) -> np.ndarray:
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    a = np.radians(deg)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * (K @ K)


ROT_CFG = {
    "rotation": {"ransac_iters": 500, "ransac_normal_thresh_deg": 5.0,
                 "yaw_bins": 360, "yaw_use_kmeans": False},
    "classes": {"structural": ["wall", "floor", "ceiling"],
                "match_classes": ["wall", "floor", "ceiling", "door", "window"]},
}
