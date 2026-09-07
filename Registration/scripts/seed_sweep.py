"""R10 §4-2 — 同じ入力メッシュを引き直すだけで解がどれだけ動くかを測る。

`50c34f2` での 20 seed（m3_cor_c E2 / `plane_match`）で、
$\\rho(W)$ が **0.073〜0.347 の二峰**に分かれ、**5/20 で基準変換を上回った**。
そこで R10 §4-2 は、分布だけでなく**どの解に落ちたか**を記録せよとしている。

現行 `main` で取り直す理由：
  - **GT 基準の計装**（`selfconsistency_*` / `gt_*` の分離）が入っている
  - **`is_inner` の修正**と **`inner_only` 既定**（R10 §2）が入っている
  - 参照定義が変わる前後の分布を混ぜないため、**切り替え後の条件で取り直す**

    conda activate sni-slam
    python Registration/scripts/seed_sweep.py --n-seeds 20
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from typing import Dict, List

import numpy as np
import yaml

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "Registration"))
os.chdir(REPO)

from regbim import io_utils, metrics, preprocess          # noqa: E402
from regbim.methods import get_method                     # noqa: E402
from regbim.metrics import class_inlier_ratio, decompose_sim3   # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="Registration/configs/realdata/m3_cor_c__E2.yaml")
    ap.add_argument("--n-seeds", type=int, default=20)
    ap.add_argument("--modes", nargs="*", default=["centroid", "plane_match"])
    ap.add_argument("--out", default="Registration/output/diag/seed_sweep.json")
    ap.add_argument("--inner-only", type=int, default=None,
                    help="参照を室内面に絞るか。1/0。既定は config の値")
    args = ap.parse_args()

    base = yaml.safe_load(open(args.config))
    base.setdefault("diagnostics", {})["record_yaw"] = True
    if args.inner_only is not None:
        base["reference"]["inner_only"] = bool(args.inner_only)
    thr = float(base["semantic_icp"]["max_corr_dist"])
    G = np.asarray(json.load(open(base["eval"]["t_gt_path"]))["T_gt"], dtype=np.float64)
    sG = decompose_sim3(G)[2]

    rows: List[Dict] = []
    print("config: %s / 参照 inner_only=%s"
          % (args.config, base["reference"].get("inner_only", True)))
    print("%-12s %5s %9s %9s %9s %9s %9s %8s"
          % ("mode", "seed", "rho_G", "rho_W", "回転度", "並進m", "s_W/s_G", "margin"))
    print("-" * 82)
    for mode in args.modes:
        for seed in range(args.n_seeds):
            cfg = copy.deepcopy(base)
            cfg["source"]["seed"] = seed
            cfg["reference"]["seed"] = seed
            cfg.setdefault("proposed", {})["translation_init"] = mode
            src = io_utils.load_source_cloud(cfg)
            dst = io_utils.load_reference_cloud(cfg)
            sp = preprocess.prepare(src, cfg)
            dp = preprocess.prepare(dst, cfg)
            m = get_method("proposed")
            T = m.register(src, dst, cfg)
            e = metrics.sim3_errors(T, G)
            rG = class_inlier_ratio(sp, dp, G, thr)
            rW = class_inlier_ratio(sp, dp, T, thr)
            d = getattr(m, "last_yaw_diag", None) or {}
            rows.append({
                "mode": mode, "seed": seed,
                "T": np.asarray(T, dtype=np.float64).tolist(),
                "rho_G": rG, "rho_W": rW, "inversion": bool(rW > rG),
                "gt_rot_deg": e["rot_deg"], "gt_trans": e["trans"],
                "gt_scale_ratio": e["scale_ratio"],
                "signed_scale": float(decompose_sim3(T)[2] / sG),
                "yaw_winner": d.get("winner"), "yaw_margin": d.get("margin"),
                "yaw_candidate_scores": d.get("candidate_scores"),
                # 失敗の3分類（R11 §3）に必要。候補ごとの回転と ICP 後の解が無いと、
                # 「候補が無い／収束しない／順位で負けた」を区別できない
                "yaw_candidate_R": d.get("candidate_R"),
                "yaw_candidate_T": d.get("candidate_T"),
                "n_seeds_scored": d.get("n_seeds_scored"),
            })
            print("%-12s %5d %9.5f %9.5f %9.2f %9.3f %9.4f %8s"
                  % (mode, seed, rG, rW, e["rot_deg"], e["trans"],
                     rows[-1]["signed_scale"],
                     "%.5f" % d["margin"] if d.get("margin") is not None else "-"),
                  flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"config": args.config,
                   "inner_only": base["reference"].get("inner_only", True),
                   "rows": rows}, f, indent=2)

    print("\n=== まとめ ===")
    for mode in args.modes:
        r = [x for x in rows if x["mode"] == mode]
        if not r:
            continue
        w = np.array([x["rho_W"] for x in r])
        inv = int(sum(x["inversion"] for x in r))
        n = len(r)
        p = inv / n
        z = 1.96
        den = 1 + z * z / n
        c = (p + z * z / (2 * n)) / den
        h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
        print("%-12s rho_W 中央値 %.5f / 範囲 [%.5f, %.5f] / 標準偏差 %.5f"
              % (mode, np.median(w), w.min(), w.max(), w.std()))
        print("%-12s   逆転 %d/%d = %.2f [%.2f, %.2f]" % ("", inv, n, p, c - h, c + h))
        # 解が何通りに分かれるか（並進 0.5 m 以内を同一クラスタとみなす）
        Ts = [np.array(x["T"]) for x in r]
        cl: List[List[int]] = []
        for i, Ti in enumerate(Ts):
            for g in cl:
                if np.linalg.norm(Ti[:3, 3] - Ts[g[0]][:3, 3]) < 0.5:
                    g.append(i)
                    break
            else:
                cl.append([i])
        print("%-12s   解のクラスタ数 %d（並進 0.5 m 以内を同一とみなす）: 大きさ %s"
              % ("", len(cl), sorted((len(g) for g in cl), reverse=True)))
        for g in sorted(cl, key=len, reverse=True)[:4]:
            q = r[g[0]]
            print("%-12s     n=%2d  回転 %.2f度  並進 %.2f m  s %.4f  rho %.3f〜%.3f"
                  % ("", len(g), q["gt_rot_deg"], q["gt_trans"], q["signed_scale"],
                     min(r[i]["rho_W"] for i in g), max(r[i]["rho_W"] for i in g)))
    print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
