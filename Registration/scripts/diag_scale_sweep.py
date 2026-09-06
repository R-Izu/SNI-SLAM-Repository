"""R8 §3-3 / §3-4 — なぜ誤った配置が基準変換より高く採点されるのかを切り分ける。

設計上の約束（R8 §3-3）
------------------------
- **点群は一度だけ前処理する。** 縮尺ごとの再サンプリングも voxel 化もしない。
  ラベル・採点対象・ゲートを固定したまま、変換だけを動かす
- 縮尺は **source の基準点 $p_0$ の行き先を固定して**振る：

      T_s(p) = a + s R (p - p_0),   a = G(p_0) または W(p_0)

  生の並進ベクトルを固定して s を振ると座標原点まわりの拡縮になり、
  **原点の取り方が診断結果を変えてしまう。**

採点（`class_inlier_ratio` と同一の定義）
----------------------------------------
分母 $N_C$ は「両側に存在するクラスの source 点数」で、**変換に依存しない**。
したがって $N_C\,\Delta\rho = N_{\rm enter} - N_{\rm leave}$ が厳密に成り立つはずであり、
それを検算する。

§3-4：参照→source の被説明率も出す。**誤答が高得点なのは参照を取り残しているからか。**

    conda activate sni-slam
    python Registration/scripts/diag_scale_sweep.py \
        --config Registration/configs/realdata/m3_cor_c__E2.yaml --w Registration/output/diag/W_m3_cor_c__E2.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np
import yaml
from scipy.spatial import cKDTree

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "Registration"))
os.chdir(REPO)

from regbim import io_utils, preprocess                     # noqa: E402
from regbim.labels import CLASS_NAMES                       # noqa: E402
from regbim.metrics import decompose_sim3                   # noqa: E402


def score_detail(src, dst, T, thresh) -> Dict:
    """`class_inlier_ratio` と同じ量に加え、クラス別内訳と受理 ID を返す。"""
    moved = (src.points @ T[:3, :3].T) + T[:3, 3]
    common = sorted(set(src.present_classes()) & set(dst.present_classes()))
    total = 0
    inl = 0
    per_class = {}
    accepted = np.zeros(len(src.points), dtype=bool)
    for c in common:
        sm = src.labels == c
        dm = dst.labels == c
        if sm.sum() == 0 or dm.sum() == 0:
            continue
        tree = cKDTree(dst.points[dm])
        d, _ = tree.query(moved[sm], k=1, workers=-1)
        ok = d < thresh
        idx = np.flatnonzero(sm)
        accepted[idx[ok]] = True
        total += len(d)
        inl += int(ok.sum())
        per_class[CLASS_NAMES[c]] = {"n_src": int(sm.sum()), "n_accept": int(ok.sum()),
                                     "frac": round(float(ok.mean()), 5)}
    return {"rho": inl / max(total, 1), "n_total": total, "n_inlier": inl,
            "per_class": per_class, "accepted": accepted}


def reverse_explained(src, dst, T, thresh) -> Dict:
    """§3-4 参照→source の被説明率。**参照の点が source に説明されているか。**"""
    moved = (src.points @ T[:3, :3].T) + T[:3, 3]
    common = sorted(set(src.present_classes()) & set(dst.present_classes()))
    total = 0
    exp = 0
    per_class = {}
    for c in common:
        sm = src.labels == c
        dm = dst.labels == c
        if sm.sum() == 0 or dm.sum() == 0:
            continue
        tree = cKDTree(moved[sm])
        d, _ = tree.query(dst.points[dm], k=1, workers=-1)
        ok = d < thresh
        total += len(d)
        exp += int(ok.sum())
        per_class[CLASS_NAMES[c]] = round(float(ok.mean()), 5)
    return {"explained": exp / max(total, 1), "n_ref": total, "per_class": per_class}


def sweep(src, dst, T, thresh, p0, scales) -> List[Dict]:
    R, _, s0 = decompose_sim3(T)
    a = (p0 @ R.T) * s0 + T[:3, 3]          # 基準点の行き先を固定する
    out = []
    for s in scales:
        Ts = np.eye(4)
        Ts[:3, :3] = s * R
        Ts[:3, 3] = a - s * (R @ p0)
        d = score_detail(src, dst, Ts, thresh)
        r = reverse_explained(src, dst, Ts, thresh)
        out.append({"s": float(s), "s_rel": float(s / s0), "rho": d["rho"],
                    "n_inlier": d["n_inlier"], "n_total": d["n_total"],
                    "per_class": d["per_class"], "reverse_explained": r["explained"],
                    "accepted": d["accepted"]})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--w", required=True, help="誤答 W の 4x4 を持つ JSON")
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-steps", type=int, default=21)
    ap.add_argument("--rel-range", type=float, nargs=2, default=[0.55, 1.25])
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    thresh = float(cfg["semantic_icp"]["max_corr_dist"])
    # ★ 前処理は一度だけ。以後どの縮尺でもこの点群を使う
    src = preprocess.prepare(io_utils.load_source_cloud(cfg), cfg)
    dst = preprocess.prepare(io_utils.load_reference_cloud(cfg), cfg)
    G = np.asarray(json.load(open(cfg["eval"]["t_gt_path"]))["T_gt"], dtype=np.float64)
    W = np.asarray(json.load(open(args.w))["T"], dtype=np.float64).reshape(4, 4)
    p0 = src.points.mean(axis=0)            # 基準点は source の重心

    print("config      : %s" % args.config)
    print("gate        : %.3f m / 前処理は1回のみ / 点数 src %d, ref %d"
          % (thresh, len(src.points), len(dst.points)))
    for name, T in (("G(基準変換)", G), ("W(誤答)", W)):
        d = score_detail(src, dst, T, thresh)
        r = reverse_explained(src, dst, T, thresh)
        _, _, s = decompose_sim3(T)
        print("\n%s  s=%.4f  rho=%.5f (%d/%d)  参照被説明率=%.5f"
              % (name, s, d["rho"], d["n_inlier"], d["n_total"], r["explained"]))
        print("  クラス別 受理率: %s"
              % {k: v["frac"] for k, v in d["per_class"].items()})
        print("  クラス別 参照被説明率: %s" % r["per_class"])

    sG = decompose_sim3(G)[2]
    scales = np.linspace(args.rel_range[0], args.rel_range[1], args.n_steps) * sG
    res = {}
    print("\n【§3-3】縮尺だけの1次元介入（R と p0 の行き先を固定して s を振る）")
    print("%-6s %8s %10s %10s %12s %12s" % ("", "s/sG", "rho(G)", "rho(W)",
                                            "参照被説明(G)", "参照被説明(W)"))
    print("-" * 64)
    swG = sweep(src, dst, G, thresh, p0, scales)
    swW = sweep(src, dst, W, thresh, p0, scales)
    for a_, b_ in zip(swG, swW):
        mark = "  <-sG" if abs(a_["s_rel"] - 1.0) < 1e-9 else ""
        print("%-6s %8.3f %10.5f %10.5f %12.5f %12.5f%s"
              % ("", a_["s_rel"], a_["rho"], b_["rho"],
                 a_["reverse_explained"], b_["reverse_explained"], mark))

    # --- 検算：N_C * Δrho = N_enter - N_leave -------------------------------
    print("\n【検算】N_C・Δrho = N_enter − N_leave（分母は変換に依存しないので厳密に成立するはず）")
    for nm, sw in (("G", swG), ("W", swW)):
        a0, a1 = sw[0], sw[-1]
        enter = int((~a0["accepted"] & a1["accepted"]).sum())
        leave = int((a0["accepted"] & ~a1["accepted"]).sum())
        lhs = a0["n_total"] * (a1["rho"] - a0["rho"])
        print("  %s: N_C・Δrho = %+.3f / N_enter − N_leave = %d − %d = %+d"
              % (nm, lhs, enter, leave, enter - leave))

    res = {"config": args.config, "gate": thresh,
           "G": {"s": float(decompose_sim3(G)[2]),
                 **{k: v for k, v in score_detail(src, dst, G, thresh).items()
                    if k != "accepted"},
                 "reverse": reverse_explained(src, dst, G, thresh)},
           "W": {"s": float(decompose_sim3(W)[2]),
                 **{k: v for k, v in score_detail(src, dst, W, thresh).items()
                    if k != "accepted"},
                 "reverse": reverse_explained(src, dst, W, thresh)},
           "sweep": [{k: v for k, v in r.items() if k != "accepted"}
                     for r in swG],
           "sweep_W": [{k: v for k, v in r.items() if k != "accepted"}
                       for r in swW]}
    out = args.out or ("Registration/output/diag/sweep_%s.json"
                       % os.path.basename(args.config)[:-5])
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print("\nwrote %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
