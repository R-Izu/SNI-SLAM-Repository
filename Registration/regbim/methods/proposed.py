"""Proposed method: physical-constraint rotation (6->3 DOF) + multi-class semantic
ICP, with a two-stage plane-constrained translation/scale initialisation.

1. Estimate the relative rotation from labels alone (floor/ceiling -> gravity,
   wall -> Manhattan yaw). This fixes 3 of the 6 rotational/translational DOF and
   leaves the 4-fold Manhattan yaw ambiguity.

2. Stage 1 -- lock yaw + scale from structural planes. For each yaw candidate,
   seed the translation and isotropic scale from the per-axis structural extents
   in the reference frame (floor+ceiling bound the vertical extent -> room height
   -> scale + height offset; walls bound the two horizontal extents -> room
   footprint -> xy scale + offset), then run rotation-fixed semantic ICP. The
   yaw whose run has the best semantic inlier ratio wins. The plane extents make
   each yaw candidate's init distinct, which is what disambiguates the Manhattan
   symmetry, and the extent ratios give a scale-aware start so the alignment
   survives a wrong initial scale.

3. Stage 2 -- refine the translation. The wall extents that disambiguate yaw also
   bias the translation when the source only partially covers a wall (the SLAM
   cloud vs the complete reference). So, holding the locked rotation and scale,
   re-seed the translation from the structural centroid and run one more
   rotation-fixed ICP, then keep whichever of {stage-1 plane result, centroid
   refinement} has the lower chamfer. Both share the locked (R, s) and differ
   only in the translation basin, so this chamfer tie-break is safe -- it keeps
   the centroid refinement when it tightens the fit and falls back to the plane
   result otherwise.

Correspondences throughout are class-constrained (semantic ICP, novelty 2): a
point only ever matches a same-class point, which lets the alignment survive
repetitive indoor structure and the asymmetry between the SLAM cloud and the
structural-only reference.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .. import plane_match, preprocess, rotation
from ..labels import NAME_TO_ID, LabeledCloud
from ..metrics import (apply_sim3, chamfer_distance, class_inlier_ratio,
                       decompose_sim3, make_sim3)
from ..semantic_icp import semantic_icp
from ..trace import Tracer
from .base import BaseRegistration
from . import register_method

# Robust span percentiles: trims a few % of stragglers (mislabelled points,
# edge normal-estimation noise) before reading a structural extent.
_LO, _HI = 2.0, 98.0
# An axis informs scale only if its structural span is room-sized rather than
# label noise (metres).
_MIN_EXTENT = 0.5
# Guard a degenerate extent ratio from injecting a wild init scale.
_SCALE_CLIP = (0.3, 3.0)


def _axis_span(points: np.ndarray, labels: np.ndarray, class_ids: List[int],
               axis: np.ndarray) -> Optional[Tuple[float, float]]:
    """Robust (center, extent) of the given classes projected onto ``axis``."""
    mask = np.isin(labels, class_ids)
    if mask.sum() < 10:
        return None
    proj = points[mask] @ axis
    lo, hi = np.percentile(proj, [_LO, _HI])
    return 0.5 * (lo + hi), float(hi - lo)


def seed_scale(src_span: Dict, dst_span: Dict, axes, scale_init: str) -> float:
    """Initial isotropic scale from per-axis structural extents.

    Split out of ``_plane_seed`` so the partial-coverage study (R3 T3) can record
    the seeded scale without re-implementing it -- a second copy would drift from
    this one, and the study would then be measuring the copy.

    ``median_axes`` (default, unchanged)
        Median of all informative per-axis extent ratios.

    ``vertical_only``
        Only the up axis (floor/ceiling span = room height). When the reference
        is a BIM of one or two rooms while the source is a SLAM map covering
        those rooms *plus* a corridor, the two horizontal extents disagree by
        2-3x and the vertical one does not: a storey's height is invariant to how
        much of the floor plan the reference covers. A median over three axes
        then picks a wrong axis two times out of three. The cost is that scale
        rests on a single scalar, so floor/ceiling noise goes straight into it
        -- quantified in T3.
    """
    if scale_init == "vertical_only":
        use = {"up"}
    elif scale_init == "median_axes":
        use = {k for k, _, _ in axes}
    else:
        raise ValueError("unknown proposed.scale_init: %r" % scale_init)

    ratios = [dst_span[k][1] / src_span[k][1]
              for k, _, _ in axes
              if k in use and dst_span[k] is not None and src_span[k] is not None
              and src_span[k][1] > _MIN_EXTENT and dst_span[k][1] > _MIN_EXTENT]
    # Same fallback as before (1.0) when no axis is informative. Deliberately NOT
    # falling back to median_axes: that would silently reinstate the behaviour
    # this option exists to avoid, and hide it from the report.
    return float(np.clip(np.median(ratios), *_SCALE_CLIP)) if ratios else 1.0


def canonical_axes(dst_p: LabeledCloud, cfg: Dict):
    """Reference canonical axes as ``[(key, unit_vector, class_ids)]``.

    Rows are wall-x, wall-y, up -- the frame each rotation candidate maps the
    source into. Shared with R3 T3 for the same reason as ``seed_scale``.
    """
    Rr = rotation.canonical_rotation(dst_p, cfg)
    wall = [NAME_TO_ID["wall"]]
    floor_ceiling = [NAME_TO_ID["floor"], NAME_TO_ID["ceiling"]]
    return [("x", Rr[0], wall), ("y", Rr[1], wall), ("up", Rr[2], floor_ceiling)]


def axis_spans(points: np.ndarray, labels: np.ndarray, axes) -> Dict:
    """``{key: (center, extent)}`` for the given canonical axes."""
    return {k: _axis_span(points, labels, ids, e) for k, e, ids in axes}


def _translation_candidates(init_T: np.ndarray, src_rot: np.ndarray,
                            labels: np.ndarray, dst_p: LabeledCloud,
                            axes, pcfg: Dict) -> List[np.ndarray]:
    """Alternative seeds whose translation comes from matching wall positions.

    ``init_T`` is the existing centroid/extent seed; its rotation and scale are
    kept and only the translation is replaced, so this changes one thing at a
    time. The vertical component uses the floor plane (a single peak), the two
    horizontal components use the 1-D wall profile correlation.

    Returns the *extra* candidates only -- ``init_T`` itself is already in the
    list built by the caller.
    """
    R, t0, s = decompose_sim3(init_T)
    wall_id, floor_id = NAME_TO_ID["wall"], NAME_TO_ID["floor"]
    src_wall = src_rot[labels == wall_id]
    dst_wall = dst_p.points[dst_p.labels == wall_id]
    src_floor = src_rot[labels == floor_id]
    dst_floor = dst_p.points[dst_p.labels == floor_id]

    bin_m = float(pcfg.get("plane_match_bin_m", 0.05))
    top_k = int(pcfg.get("plane_match_top_k", 3))
    max_shift = float(pcfg.get("plane_match_max_shift_m", 30.0))

    # Vertical: floors give one sharp peak, so take the plane offset directly.
    up = axes[2][1]
    dz = None
    if len(src_floor) >= 10 and len(dst_floor) >= 10:
        dz = plane_match.plane_offset(s * (src_floor @ up), dst_floor @ up)

    # Horizontal: correlate the wall profiles on each canonical axis.
    per_axis: List[List[float]] = []
    for key, e, _ in axes[:2]:
        if len(src_wall) < 10 or len(dst_wall) < 10:
            per_axis.append([])
            continue
        cands = plane_match.offset_candidates(
            s * (src_wall @ e), dst_wall @ e, bin_m=bin_m, top_k=top_k,
            max_shift_m=max_shift)
        per_axis.append([d for d, _ in cands])

    # `offset_candidates` returns d with the meaning "s*(src . e) + d lines up with
    # dst . e", so d IS the new translation component along e -- it replaces the
    # old one rather than being added to it.
    out: List[np.ndarray] = []
    for dx in (per_axis[0] or [None]):
        for dy in (per_axis[1] or [None]):
            if dx is None and dy is None and dz is None:
                continue
            t = t0.copy()
            for comp, e in ((dx, axes[0][1]), (dy, axes[1][1]), (dz, up)):
                if comp is not None:
                    t = t - (t @ e) * e + comp * e
            T = make_sim3(R, t, s)
            if not np.allclose(T, init_T):
                out.append(T)
    return out


def _struct_centroid(points: np.ndarray, labels: np.ndarray,
                     struct_ids: List[int]) -> np.ndarray:
    mask = np.isin(labels, struct_ids)
    return points[mask].mean(axis=0) if mask.any() else points.mean(axis=0)


def _chamfer(T: np.ndarray, src: LabeledCloud, dst: LabeledCloud) -> float:
    return chamfer_distance(apply_sim3(T, src.points), dst.points)


@register_method("proposed")
class Proposed(BaseRegistration):
    name = "proposed"

    # Filled by ``register`` only when ``diagnostics.record_yaw`` is on; read by
    # the caller right after the call. Not used by the method itself.
    last_yaw_diag: Optional[Dict] = None

    def register(self, src: LabeledCloud, dst: LabeledCloud, cfg: Dict,
                 tracer: Optional[Tracer] = None) -> np.ndarray:
        src_p = preprocess.prepare(src, cfg)
        dst_p = preprocess.prepare(dst, cfg)
        thresh = float(cfg["semantic_icp"]["max_corr_dist"])
        abl = cfg.get("ablation") or {}

        struct_ids = [NAME_TO_ID[n] for n in cfg["classes"]["structural"]]

        if abl.get("no_gravity"):
            # Ablation `no_gravity` = **removal of the entire physical-constraint
            # stage**, not "the contribution of gravity alone": gravity
            # canonicalisation, Manhattan yaw candidates and plane-extent seeding
            # all go together, because they all derive from the canonical frame.
            # Do not report it as isolating gravity (R7 §7).
            # Init = structural-centroid alignment at unit scale; rotation left to
            # free Sim3 semantic ICP.
            c_src = _struct_centroid(src_p.points, src_p.labels, struct_ids)
            c_dst = _struct_centroid(dst_p.points, dst_p.labels, struct_ids)
            init_T = make_sim3(np.eye(3), c_dst - c_src, 1.0)
            return semantic_icp(src_p, dst_p, init_T, cfg, rotation_fixed=False,
                                tracer=tracer)

        # Ablation `single_class` = **removal of class-constrained correspondence**
        # (ICP matching and the candidate score stop distinguishing classes).
        # It is NOT "no semantic information": labels still drive the gravity /
        # Manhattan estimation, the per-class extents and the structural-point
        # selection below. Do not report it as a semantics ablation (R7 §7).
        if abl.get("single_class"):
            match_ids = [NAME_TO_ID[n] for n in cfg["classes"]["match_classes"]]
            one = match_ids[0]
            src_score = LabeledCloud(
                src_p.points, np.where(np.isin(src_p.labels, match_ids), one, 0),
                src_p.normals)
            dst_score = LabeledCloud(
                dst_p.points, np.where(np.isin(dst_p.labels, match_ids), one, 0),
                dst_p.normals)
        else:
            src_score, dst_score = src_p, dst_p

        axes = canonical_axes(dst_p, cfg)
        dst_span = axis_spans(dst_p.points, dst_p.labels, axes)
        c_dst = _struct_centroid(dst_p.points, dst_p.labels, struct_ids)

        # --- stage 1: plane-seeded search -> reliable yaw R and scale s ----------
        candidates: List[np.ndarray] = rotation.relative_rotation_candidates(
            src_p, dst_p, cfg)
        # R5 §2-5 / Q3: record which Manhattan candidate won and by how much.
        # T3 showed the failure mode is picking the *wrong* candidate (66% of
        # failures involve rotation, worst excess 175 deg), so the scores that
        # decide it have to be observable. Opt-in via config so every existing
        # config produces byte-identical output; the method itself is unchanged.
        record_yaw = bool((cfg.get("diagnostics") or {}).get("record_yaw", False))
        pcfg = cfg.get("proposed") or {}
        scale_init = str(pcfg.get("scale_init", "median_axes"))
        translation_init = str(pcfg.get("translation_init", "centroid"))
        cand_scores: List[float] = []
        cand_T: List[List[List[float]]] = []
        plane_T = None
        plane_trace: Optional[Tracer] = None
        best_score = -np.inf

        # Build the pose candidates. With `centroid` there is exactly one per yaw,
        # which is the original behaviour; `plane_match` may add a few translation
        # candidates per yaw (see _translation_candidates).
        seeds: List[Tuple[np.ndarray, np.ndarray]] = []      # (init_T, R)
        for R in candidates:
            src_rot = src_p.points @ R.T                   # source in reference frame
            c_src = _struct_centroid(src_rot, src_p.labels, struct_ids)
            init_T = self._plane_seed(src_rot, src_p.labels, R, axes, dst_span,
                                      c_dst, c_src,
                                      fixed_scale=bool(abl.get("fixed_scale")),
                                      scale_init=scale_init)
            seeds.append((init_T, R))
            if translation_init == "plane_match":
                seeds.extend(
                    (T_alt, R) for T_alt in
                    _translation_candidates(init_T, src_rot, src_p.labels,
                                            dst_p, axes, pcfg))
            elif translation_init != "centroid":
                raise ValueError("unknown proposed.translation_init: %r"
                                 % translation_init)

        if translation_init == "plane_match":
            # Scoring every seed with a full ICP would multiply the cost by the
            # number of candidates. Rank them at the seed pose first (no ICP) and
            # only run ICP on the best few. The `centroid` path skips this and runs
            # ICP on its single seed per yaw, exactly as before.
            keep = int(pcfg.get("max_icp_candidates", 8))
            seeds = sorted(
                seeds,
                key=lambda s: -class_inlier_ratio(src_score, dst_score, s[0], thresh)
            )[:max(keep, 1)]

        cand_R: List[np.ndarray] = []
        for init_T, R in seeds:
            cand_R.append(R)
            # Each candidate gets its own tracer; only the winner's trajectory
            # is surfaced, so the reported curve is a single coherent ICP run.
            cand_tracer = Tracer(tracer.stride) if tracer is not None else None
            T = semantic_icp(src_p, dst_p, init_T, cfg, rotation_fixed=True,
                             tracer=cand_tracer)
            score = class_inlier_ratio(src_score, dst_score, T, thresh)
            if record_yaw:
                cand_scores.append(float(score))
                # Each candidate's post-ICP pose, not just its score. From the
                # winner alone, "the right candidate was never generated", "it was
                # generated but did not converge" and "it converged but lost on
                # score" cannot be told apart (R11 §3).
                cand_T.append(np.asarray(T, dtype=np.float64).tolist())
            if score > best_score:
                best_score = score
                plane_T = T
                plane_trace = cand_tracer

        if record_yaw:
            order = np.argsort(cand_scores)[::-1]
            self.last_yaw_diag = {
                # 候補ごとの rho_k。**差が小さいときに間違えているか**を見るための量
                "candidate_scores": [round(s, 5) for s in cand_scores],
                "winner": int(np.argmax(cand_scores)),
                # 1位と2位の差。小さいほど「たまたま選んだ」に近い
                "margin": (round(cand_scores[order[0]] - cand_scores[order[1]], 5)
                           if len(cand_scores) > 1 else None),
                # 正解の候補を決めるのは呼び出し側（期待する回転を知っているのは評価側）。
                # ここでは候補の回転そのものを渡す。**`candidate_scores` と同じ順序**
                # であることが必要で、plane_match では1つのヨーに複数の並進候補が
                # 対応するので、ヨーの一覧ではなく**採点した順**に並べる。
                "candidate_R": [R.tolist() for R in cand_R],
                "candidate_T": cand_T,
                "translation_init": translation_init,
                "n_seeds_scored": len(cand_scores),
            }

        # --- stage 2: centroid-refine translation at the locked (R, s) -----------
        R_win, _, s_win = decompose_sim3(plane_T)
        c_src_win = _struct_centroid(src_p.points @ R_win.T, src_p.labels, struct_ids)
        refine_T0 = make_sim3(R_win, c_dst - s_win * c_src_win, s_win)
        refine_trace = Tracer(tracer.stride) if tracer is not None else None
        refine_T = semantic_icp(src_p, dst_p, refine_T0, cfg, rotation_fixed=True,
                                tracer=refine_trace)

        # Pick by chamfer: safe here because both poses share the locked (R, s)
        # and differ only in translation basin (no cross-yaw/scale ambiguity).
        cand = [(_chamfer(plane_T, src_p, dst_p), plane_T, plane_trace),
                (_chamfer(refine_T, src_p, dst_p), refine_T, refine_trace)]
        _, win_T, win_trace = min(cand, key=lambda x: x[0])
        if tracer is not None and win_trace is not None:
            tracer.steps = win_trace.steps
        return win_T

    @staticmethod
    def _plane_seed(src_rot: np.ndarray, labels: np.ndarray, R: np.ndarray, axes,
                    dst_span, c_dst: np.ndarray, c_src: np.ndarray,
                    fixed_scale: bool = False,
                    scale_init: str = "median_axes") -> np.ndarray:
        """Scale + translation from per-axis structural extents (rotation fixed)."""
        src_span = axis_spans(src_rot, labels, axes)
        s = seed_scale(src_span, dst_span, axes, scale_init)
        if fixed_scale:
            s = 1.0    # ablation: GT scale is pre-applied upstream (rigid mode)

        # Centroid alignment at the estimated scale, then overwrite each axis
        # component with its plane-matched offset where available (guarantees a
        # full vector even if an axis is missing).
        t = c_dst - s * c_src
        for k, e, _ in axes:
            if dst_span[k] is not None and src_span[k] is not None:
                cd, cs = dst_span[k][0], src_span[k][0]
                t = t - (t @ e) * e + (cd - s * cs) * e
        return make_sim3(R, t, s)
