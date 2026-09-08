"""R14 §3-1 / R15 §3：保存済みの全候補を、同一条件で双方向に採点する。

**手法は再実行しない。** `sweep_*_io{0,1}.json` に保存された `yaw_candidate_T`
（160 実行 × 4〜6 候補）をそのまま採点する。

| 量 | 定義 | 分母 |
|---|---|---|
| rho_P | source 点のうち、同一クラスの参照点がゲート内にある割合 | **source 全点** |
| rho_Q | **Q_room** の点のうち、同一クラスの source 点がゲート内にある割合 | **Q_room 全点** |

**どちらの分母からも、相手側に無いクラスを消さない**（R14 §3-2）。
`regbim.metrics.class_inlier_ratio` は共通クラスだけを分母にしているので使えない。
**手法の目的関数は変えないので、ここに別実装を置く。**

$Q_{\rm room}$ は対象室の室内向き面（IFC の `is_inner`）。**GT は使わない。**
**`io0`（両面参照）の候補も、同じ $Q_{\rm room}$ で採点する**（R14 §3-2「分母は固定」）。
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import yaml

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "Registration"))
sys.path.insert(0, os.path.join(REPO, "Registration", "scripts"))
os.chdir(REPO)

from failure_decomposition import classify, provenance        # noqa: E402
from regbim import io_utils                                   # noqa: E402
from regbim.metrics import (apply_sim3, decompose_sim3,       # noqa: E402
                            rotation_error_deg)


def explained_ratio(a_pts: np.ndarray, a_lab: np.ndarray,
                    b_pts: np.ndarray, b_lab: np.ndarray, thresh: float) -> float:
    """a のうち、同一クラスの b がゲート内にある割合。**分母は a の全点。**

    b に無いクラスの a 点は「説明できなかった」として分母に残る。ここが
    ``class_inlier_ratio``（共通クラスのみを分母にする）との違いである。
    """
    from scipy.spatial import cKDTree

    if len(a_pts) == 0:
        return float("nan")
    inl = 0
    for c in np.unique(a_lab):
        am = a_lab == c
        bm = b_lab == c
        if bm.sum() == 0:
            continue                      # 分母には残す。inl には足さない
        dist, _ = cKDTree(b_pts[bm]).query(a_pts[am], k=1, workers=-1)
        inl += int((dist < thresh).sum())
    return float(inl / len(a_pts))


def score_pair(src_pts, src_lab, q_pts, q_lab, T, thresh) -> Dict[str, float]:
    moved = apply_sim3(np.asarray(T, dtype=np.float64), src_pts)
    return {"rho_P": explained_ratio(moved, src_lab, q_pts, q_lab, thresh),
            "rho_Q": explained_ratio(q_pts, q_lab, moved, src_lab, thresh)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweeps", nargs="+", default=sorted(glob.glob(
        "Registration/output/diag/sweep_*_io[01].json")))
    ap.add_argument("--rot-tol", type=float, default=5.0)
    ap.add_argument("--basin", type=float, default=1.0)
    ap.add_argument("--out", default="Registration/output/diag/bidirectional_scores.json")
    args = ap.parse_args()

    recs: List[Dict] = []
    for path in args.sweeps:
        blob = json.load(open(path))
        rows = blob.get("rows") or []
        if not rows or "yaw_candidate_T" not in rows[0]:
            print("skip (候補行列なし): %s" % path)
            continue
        cfg = yaml.safe_load(open(blob["config"]))
        G = np.asarray(json.load(open(cfg["eval"]["t_gt_path"]))["T_gt"],
                       dtype=np.float64)
        thresh = float(cfg["semantic_icp"]["max_corr_dist"])
        src = io_utils.load_source_cloud(cfg)
        # **Q_room は固定**：io0（両面）の候補も室内面で採点する
        q_cfg = dict(cfg, reference=dict(cfg["reference"], inner_only=True))
        qroom = io_utils.load_reference_cloud(q_cfg)
        p0 = src.points.mean(axis=0)
        gp = apply_sim3(G, p0[None])[0]
        RG, _, sG = decompose_sim3(G)
        print("=== %s ===" % os.path.basename(path))
        print("  source %d 点 / Q_room %d 点 / ゲート %.2f m / s_G=%.4f"
              % (len(src), len(qroom), thresh, sG))
        print("  Q_room のクラス: %s" % sorted(set(qroom.labels.tolist())))
        print("  source のクラス: %s" % sorted(set(src.labels.tolist())))

        for row in rows:
            cls = classify(row, G, args.rot_tol, args.basin, p0)
            for idx, T in enumerate(row["yaw_candidate_T"]):
                T = np.asarray(T, dtype=np.float64)
                R, _, s = decompose_sim3(T)
                sc = score_pair(src.points, src.labels,
                                qroom.points, qroom.labels, T, thresh)
                rot = rotation_error_deg(R, RG)
                disp = float(np.linalg.norm(apply_sim3(T, p0[None])[0] - gp))
                recs.append({
                    "sweep": os.path.basename(path),
                    "inner_only": blob.get("inner_only"),
                    "mode": row["mode"], "seed": row["seed"], "cand": idx,
                    "rho_P": sc["rho_P"], "rho_Q": sc["rho_Q"],
                    "gt_rot_deg": rot, "ref_point_disp_m": disp,
                    "s": s, "s_over_sG": float(s / sG),
                    "is_correct": bool(rot < args.rot_tol and disp < args.basin),
                    "is_winner": bool(row.get("yaw_winner") == idx),
                    "run_class": cls["class"],
                    "method_score": (row.get("yaw_candidate_scores") or [None] * 9)[idx]
                    if idx < len(row.get("yaw_candidate_scores") or []) else None,
                })
            print("  seed %-3s %-12s %s" % (row["seed"], row["mode"], cls["class"]),
                  flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"provenance": provenance(), "rot_tol": args.rot_tol,
                   "basin": args.basin, "records": recs}, f, indent=2)
    print("\n候補 %d 件を採点した → %s" % (len(recs), args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
