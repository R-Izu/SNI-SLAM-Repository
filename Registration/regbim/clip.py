"""Partial-coverage clipping of the reference cloud (R3 T3).

Why this exists
---------------
In the real setting the BIM covers one or two rooms while the SLAM map covers
those rooms *plus* a corridor and an opposite room. Nothing in the existing
evaluation exercises that: reference and source always cover the same space.
Clipping only the reference on Replica reproduces the condition on data where
the ground truth is known, **before** committing to it on real data.

Coverage is defined by **floor area**, not by bounding-box side length: the
instruction asks for the achieved value to be reported because the nominal one
and the real one differ (an L-shaped room loses area faster than its bbox does).
Area is measured by XY occupancy cells rather than a convex hull -- a hull would
fill in exactly the concavities that make a floor plan distinctive.

A config without ``reference.clip`` is untouched, so every existing result is
reproduced bit for bit.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from .labels import NAME_TO_ID, LabeledCloud

_CELL = 0.1          # metres; XY occupancy cell for the area measurement


def _occupied_cells(xy: np.ndarray) -> np.ndarray:
    return np.unique(np.floor(xy / _CELL).astype(np.int64), axis=0)


def floor_area_m2(cloud: LabeledCloud) -> float:
    """Occupied floor area, from the points labelled floor (fallback: all)."""
    m = cloud.labels == NAME_TO_ID["floor"]
    xy = cloud.points[m][:, :2] if m.sum() >= 10 else cloud.points[:, :2]
    return len(_occupied_cells(xy)) * _CELL * _CELL


def _area_inside(cloud: LabeledCloud, lo: np.ndarray, hi: np.ndarray) -> float:
    m = cloud.labels == NAME_TO_ID["floor"]
    xy = cloud.points[m][:, :2] if m.sum() >= 10 else cloud.points[:, :2]
    k = (xy >= lo).all(axis=1) & (xy <= hi).all(axis=1)
    if not k.any():
        return 0.0
    return len(_occupied_cells(xy[k])) * _CELL * _CELL


def clip_reference(cloud: LabeledCloud, spec: Optional[Dict]) -> LabeledCloud:
    """Clip ``cloud`` to an XY box holding ``spec['keep_frac']`` of its floor area.

    ``spec``:
      ``keep_frac``  target fraction of floor area to keep (1.0 -> no clipping)
      ``anchor``     ``[ax, ay]`` with each component in ``{-1, 0, +1}``: which
                     edge the kept box is aligned to (-1 low, +1 high, 0 centre).
                     Several anchors give the spread the instruction asks for.

    Why the box is anchored to an edge, not centred
    -----------------------------------------------
    A centred box that holds a fraction of the *floor* sits strictly inside the
    room, and the walls are on its perimeter -- so it drops **every wall** and
    the yaw estimator fails outright ("need wall normals to estimate yaw").
    Measured on Replica ``room_0``: at 75% coverage a centred box left 2,839 wall
    points of 85,725 and at 50% it left none, while floor and ceiling survived
    intact. That is an artefact of the clipping, not the condition being
    modelled.

    The real condition is that the BIM covers **fewer complete rooms** than the
    SLAM map -- the retained part still has its own real exterior walls. On a
    single-room scene the closest equivalent is to push the kept box against a
    corner, so the two walls on that side are retained in full and the cut runs
    through the interior on the other two. That is what an anchor of
    ``[-1, -1]`` does; the four corners give the positional spread.

    One scalar ``f`` shrinks both sides together, found by bisection on the
    achieved area rather than a closed form, because the achieved area depends
    on where the floor actually is -- which is the whole point of measuring it.
    """
    if not spec:
        return cloud
    keep = float(spec.get("keep_frac", 1.0))
    if keep >= 1.0:
        return cloud

    p = cloud.points
    lo_all, hi_all = p[:, :2].min(axis=0), p[:, :2].max(axis=0)
    ext = hi_all - lo_all
    anchor = np.array([float(v) for v in (spec.get("anchor") or [0.0, 0.0])])

    total = floor_area_m2(cloud)
    want = keep * total

    def box(f: float) -> Tuple[np.ndarray, np.ndarray]:
        side = f * ext
        # anchor -1 -> flush with the low edge, +1 -> the high edge, 0 -> centred
        lo = lo_all + (1.0 + anchor) * 0.5 * (ext - side)
        return lo, lo + side

    f_lo, f_hi = 0.0, 1.0                 # 1.0 is the whole cloud
    for _ in range(40):
        f = 0.5 * (f_lo + f_hi)
        if _area_inside(cloud, *box(f)) < want:
            f_lo = f
        else:
            f_hi = f
    lo, hi = box(0.5 * (f_lo + f_hi))

    keep_mask = (p[:, :2] >= lo).all(axis=1) & (p[:, :2] <= hi).all(axis=1)
    if keep_mask.sum() < 100:
        raise ValueError("clip left %d points; keep_frac=%.2f anchor=%s"
                         % (keep_mask.sum(), keep, anchor.tolist()))
    out = cloud.subset(keep_mask)
    achieved = floor_area_m2(out)
    out.meta = dict(cloud.meta)
    out.meta["clip"] = {
        "keep_frac_nominal": keep, "anchor": [float(v) for v in anchor],
        "class_counts_after": {int(c): int((out.labels == c).sum())
                               for c in np.unique(out.labels)},
        "box_min_xy": [round(float(v), 3) for v in lo],
        "box_max_xy": [round(float(v), 3) for v in hi],
        "floor_area_full_m2": round(total, 2),
        "floor_area_clipped_m2": round(achieved, 2),
        # ★ the number to report: nominal and achieved differ
        "coverage_achieved": round(achieved / total, 4) if total > 0 else None,
        "n_points_before": int(len(cloud)), "n_points_after": int(len(out)),
    }
    return out
