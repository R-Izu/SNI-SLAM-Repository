"""Coverage-invariant translation seeding by 1-D histogram matching.

Why
---
The proposed method seeds the translation from the *structural centroid* and from
the *centre of the per-axis extent* (``_axis_span``). Both are statistics of the
observed range, so both move when the reference covers only part of the source:

    reference = BIM of one room, source = SLAM map of that room + a corridor
    -> the two centroids are several metres apart, and the seed inherits that gap.

A plane's position along its own normal does **not** move when you observe only
half of it. Matching the *positions of the walls* instead of the *centre of the
span* therefore removes the coverage dependence.

Why cross-correlation and not "align the strongest peak"
--------------------------------------------------------
Under partial coverage the two sides do not have the same set of walls: the
source sees corridor walls the reference does not contain. Aligning the single
strongest peak can therefore align two *different* walls. Correlating the whole
1-D profile lets the remaining walls outvote a missing one.

Scope
-----
This module only proposes candidate offsets. It does not decide anything: the
caller scores the resulting poses with the same criterion it already uses for the
Manhattan yaw candidates.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


def _histogram(x: np.ndarray, lo: float, n_bins: int, bin_m: float) -> np.ndarray:
    h, _ = np.histogram(x, bins=n_bins, range=(lo, lo + n_bins * bin_m))
    total = h.sum()
    # Normalised so that a side with more points does not dominate the product.
    return h.astype(np.float64) / (total if total else 1.0)


def offset_candidates(src_proj: np.ndarray, dst_proj: np.ndarray,
                      bin_m: float = 0.05, top_k: int = 3,
                      max_shift_m: float = 30.0,
                      min_separation_m: float = 0.5) -> List[Tuple[float, float]]:
    """Offsets ``d`` such that ``src_proj + d`` best lines up with ``dst_proj``.

    Both inputs are 1-D coordinates along one axis (the source already scaled).
    Returns ``[(offset_m, score)]``, best first, at most ``top_k`` entries and at
    least ``min_separation_m`` apart so that neighbouring bins of one peak do not
    fill the list.
    """
    if len(src_proj) < 10 or len(dst_proj) < 10:
        return []
    lo = float(min(src_proj.min(), dst_proj.min()) - max_shift_m)
    hi = float(max(src_proj.max(), dst_proj.max()) + max_shift_m)
    n_bins = int(np.ceil((hi - lo) / bin_m))
    if n_bins < 4 or n_bins > 200000:          # degenerate or absurdly large
        return []
    hs = _histogram(src_proj, lo, n_bins, bin_m)
    hd = _histogram(dst_proj, lo, n_bins, bin_m)

    # corr[k] is the overlap when the source profile is shifted right by
    # (k - (n_bins - 1)) bins.
    corr = np.correlate(hd, hs, mode="full")
    shifts = (np.arange(len(corr)) - (n_bins - 1)) * bin_m
    ok = np.abs(shifts) <= max_shift_m
    corr, shifts = corr[ok], shifts[ok]
    if not len(corr) or corr.max() <= 0:
        return []

    out: List[Tuple[float, float]] = []
    order = np.argsort(corr)[::-1]
    for i in order:
        d = float(shifts[i])
        if all(abs(d - p) >= min_separation_m for p, _ in out):
            out.append((d, float(corr[i])))
        if len(out) >= top_k:
            break
    return out


def plane_offset(src_proj: np.ndarray, dst_proj: np.ndarray,
                 bin_m: float = 0.02) -> float:
    """Offset that lines up the two dominant planes (used for the vertical axis).

    Floors give a single sharp peak, so the full correlation buys nothing here and
    the mode is both cheaper and easier to reason about.
    """
    def mode(x: np.ndarray) -> float:
        lo, hi = float(x.min()), float(x.max())
        n = max(int(np.ceil((hi - lo) / bin_m)), 1)
        h, edges = np.histogram(x, bins=n, range=(lo, lo + n * bin_m))
        k = int(np.argmax(h))
        return float(0.5 * (edges[k] + edges[k + 1]))

    return mode(dst_proj) - mode(src_proj)
