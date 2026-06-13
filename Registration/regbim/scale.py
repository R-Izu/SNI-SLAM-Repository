"""Closed-form similarity (Sim3) estimation from point correspondences.

``umeyama`` solves the full 7-DOF similarity; ``scale_translation`` solves only
the 4 remaining DOF (scale + translation) when rotation is already fixed by the
physical-constraint step.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def umeyama(
    src: np.ndarray,
    dst: np.ndarray,
    weights: Optional[np.ndarray] = None,
    with_scaling: bool = True,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Weighted Umeyama: find R, t, s minimising sum w |dst - (s R src + t)|^2.

    Returns ``(R (3x3), t (3,), s)``. Handles the reflection case so ``det(R)=+1``.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n = len(src)
    if n < 3:
        raise ValueError("need >= 3 correspondences for Umeyama")
    if weights is None:
        w = np.ones(n)
    else:
        w = np.asarray(weights, dtype=np.float64)
    w = w / (w.sum() + 1e-12)

    mu_src = (w[:, None] * src).sum(axis=0)
    mu_dst = (w[:, None] * dst).sum(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst

    cov = (w[:, None] * dst_c).T @ src_c            # (3,3)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt

    if with_scaling:
        var_src = (w * (src_c ** 2).sum(axis=1)).sum()
        s = float((D * np.diag(S)).sum() / (var_src + 1e-12))
    else:
        s = 1.0
    t = mu_dst - s * R @ mu_src
    return R, t, s


def scale_translation(
    src: np.ndarray,
    dst: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float]:
    """With rotation fixed (R = I), solve for scale s and translation t.

    Minimises sum w |dst - (s src + t)|^2. Returns ``(t (3,), s)``.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if weights is None:
        w = np.ones(len(src))
    else:
        w = np.asarray(weights, dtype=np.float64)
    w = w / (w.sum() + 1e-12)

    mu_src = (w[:, None] * src).sum(axis=0)
    mu_dst = (w[:, None] * dst).sum(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst
    num = (w * (src_c * dst_c).sum(axis=1)).sum()
    den = (w * (src_c ** 2).sum(axis=1)).sum()
    s = float(num / (den + 1e-12))
    t = mu_dst - s * mu_src
    return t, s
