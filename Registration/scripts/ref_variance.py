"""R30 §4 手順1 — **参照の標本を変えただけで結果がどれだけ動くか**を測る。

なぜ先に測るか（R30 §4）
------------------------
`_load_replica_reference` はシード無しで表面標本化しており、**実行ごとに参照が変わる**。
シードを入れれば再現できるようになるが、**どの標本を引くかで結果がどれだけ動くかは、
それ自体が不確かさである。固定して見えなくするのではなく、一度測ってから固定する。**

この 10 実行で動いているもの
----------------------------
| | |
|---|---|
| source | `points_ply`。**決定的**（同じファイルを読むだけ） |
| 摂動列 | seed 固定。**試行 i の摂動は毎回同じ** |
| **参照点群** | **シード無し。毎回ちがう 20 万点** |

**したがって run 間で動くのは参照の標本だけである。**

成功率は飽和しうるので、**$d_\Omega$ の分布も出す**（成功率だけ見ると「差が無い」に見える）。

    python Registration/scripts/ref_variance.py
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np
import yaml

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "Registration"))
sys.path.insert(0, os.path.join(REPO, "Registration", "scripts"))
os.chdir(REPO)

from criterion_verdict import build_omega, d_omega      # noqa: E402
from failure_decomposition import provenance            # noqa: E402
from regbim import metrics                              # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="output/Registration/gt_a_refvar")
    ap.add_argument("--scene", default="room_0")
    ap.add_argument("--method", default="proposed")
    ap.add_argument("--n-runs", type=int, default=10)
    ap.add_argument("--out", default="Registration/output/diag/ref_variance.json")
    args = ap.parse_args()

    cfg = yaml.safe_load(open("Registration/configs/gt_a/%s.yaml" % args.scene))
    omega = build_omega(cfg)          # Ω は source 由来なので run 間で不変
    G = np.asarray(json.load(open(cfg["eval"]["t_gt_path"]))["T_gt"],
                   dtype=np.float64).reshape(4, 4)

    rows = []
    print("%-6s %10s %12s %12s %12s %12s"
          % ("run", "成功率", "d_Ω 中央", "d_Ω 最大", "chamfer", "試行数"))
    print("-" * 68)
    for i in range(1, args.n_runs + 1):
        d = os.path.join(args.base, "run_%d" % i)
        rc = os.path.join(d, "results.csv")
        tm = os.path.join(d, "trial_matrices.json")
        if not (os.path.exists(rc) and os.path.exists(tm)):
            print("run_%d: 出力が無い。飛ばす" % i)
            continue
        res = {r["method"]: r for r in csv.DictReader(open(rc))}[args.method]
        mats = json.load(open(tm))
        ds = []
        for t in mats["trials"]:
            if t["method"] != args.method or t["trial"] < 0:
                continue
            T = np.asarray(t["T_est"], dtype=np.float64)
            P = np.asarray(t["P"], dtype=np.float64)
            ds.append(d_omega(T, G @ metrics.invert_sim3(P), omega))
        ds = np.asarray(ds)
        row = {"run": i, "gt_success_rate": float(res["gt_success_rate"]),
               "n_trials": int(len(ds)),
               "d_omega_median": float(np.median(ds)),
               "d_omega_max": float(ds.max()),
               "d_omega_mean": float(ds.mean()),
               "direct_chamfer": float(res["direct_chamfer"])}
        rows.append(row)
        print("%-6d %10.3f %12.5f %12.5f %12.5f %12d"
              % (i, row["gt_success_rate"], row["d_omega_median"],
                 row["d_omega_max"], row["direct_chamfer"], row["n_trials"]))

    if not rows:
        print("**出力が1件も無い**")
        return 1

    print("\n## run 間のばらつき（**参照の標本だけが違う**）\n")
    for key, label, unit in (("gt_success_rate", "成功率", ""),
                             ("d_omega_median", "d_Ω の中央値", " m"),
                             ("d_omega_max", "d_Ω の最大値", " m"),
                             ("direct_chamfer", "chamfer", " m")):
        v = np.asarray([r[key] for r in rows])
        span = v.max() - v.min()
        print("  %-14s 中央 %.5f%s / 範囲 [%.5f, %.5f] / 幅 %.5f%s"
              % (label, float(np.median(v)), unit, v.min(), v.max(), span, unit))
        if key == "gt_success_rate" and span == 0:
            print("       **10 実行すべて同じ。成功率は飽和しており、"
                  "この量ではばらつきを検出できない**")

    dm = np.asarray([r["d_omega_median"] for r in rows])
    print("\n**参照の標本による d_Ω の中央値の振れ幅は %.5f m である。**"
          % (dm.max() - dm.min()))
    print("**事前登録の閾値 0.1 m に対して %.2f%% にあたる。**"
          % (100 * (dm.max() - dm.min()) / 0.1))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"provenance": provenance(), "scene": args.scene,
                   "method": args.method,
                   "what_varies": "参照点群の表面標本のみ（source と摂動列は固定）",
                   "seeded": False, "runs": rows}, f, indent=2, ensure_ascii=False)
    print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
