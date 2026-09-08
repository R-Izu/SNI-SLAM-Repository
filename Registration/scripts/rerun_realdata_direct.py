"""実データ 40 条件の無摂動解を、現在のコード・現在の参照定義で出し直す。

**なぜ必要か**：保存済みの `realdata_results.json` は 2026-09-04 の実行で、
そのとき config に `inner_only` の指定が無く、**既定の「両面参照」で走っていた**。
現在の config は 40 本すべて `inner_only: true` を指定しており、さらに
`is_inner` 自体が 2026-09-06 に修正されている（共有壁の両面を落としていた欠陥）。
**つまり保存済みの 40 件は、いまとは別の参照定義で得た値である。**

**摂動 100 試行は回さない。** その 100 試行が出すのは自己一貫性の量だけで、
**GT 基準の量は無摂動解 (`direct_*`) にしかない**（保存済みでも、
GT から 180 度ずれている条件の `success_rate` が最大 1.00 になっている）。
100 試行 × 40 条件は約 98 時間、無摂動解のみなら約 1 時間である。

**1 件ごとに書き出す**（F6：長時間ジョブは途中結果を残す）。
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import yaml

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "Registration"))
sys.path.insert(0, os.path.join(REPO, "Registration", "scripts"))
os.chdir(REPO)

from failure_decomposition import provenance                  # noqa: E402
from regbim import io_utils, metrics                          # noqa: E402
from regbim.methods import get_method                         # noqa: E402


def mode_of(deg: float) -> str:
    if deg < 5:
        return "正"
    if 60 < deg < 120:
        return "90度"
    if deg > 150:
        return "180度"
    return "他(%.0f度)" % deg


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0,
                    help="点サンプリングの seed。既定 0 で固定し、再現可能にする")
    ap.add_argument("--out", default="Registration/output/diag/realdata_direct_v2.json")
    args = ap.parse_args()

    stored = {("%s__%s" % (r["scene"], r["condition"])): r
              for r in json.load(open("Registration/output/realdata/realdata_results.json"))}
    cfgs = sorted(glob.glob("Registration/configs/realdata/*.yaml"))
    print("%d 条件。点サンプリング seed=%d 固定。" % (len(cfgs), args.seed))
    print("%-16s %8s %9s %8s  %-6s | 保存済み(両面参照)" % (
        "target", "回転", "並進", "縮尺比", "モード"))

    out = []
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    for path in cfgs:
        name = os.path.basename(path)[:-5]
        cfg = yaml.safe_load(open(path))
        cfg["source"] = dict(cfg["source"], seed=args.seed)
        G = np.asarray(json.load(open(cfg["eval"]["t_gt_path"]))["T_gt"], dtype=np.float64)
        t0 = time.time()
        T = get_method("proposed").register(io_utils.load_source_cloud(cfg),
                                            io_utils.load_reference_cloud(cfg), cfg)
        e = metrics.sim3_errors(T, G)
        succ = cfg["eval"]["success"]
        ok = (not e["degenerate"] and e["rot_deg"] < succ["rot_deg"]
              and e["trans"] < succ["trans"] and e["scale_ratio"] < succ["scale_ratio"])
        s0 = stored.get(name)
        rec = {"target": name, "scene": name.split("__")[0],
               "condition": name.split("__")[1],
               "rot_deg": e["rot_deg"], "trans": e["trans"],
               "scale_ratio": e["scale_ratio"], "degenerate": bool(e["degenerate"]),
               "mode": mode_of(e["rot_deg"]), "success": bool(ok),
               "time_s": round(time.time() - t0, 1),
               "stored": s0 and {k: s0[k] for k in
                                 ("direct_rot_deg", "direct_trans", "direct_scale_ratio")}}
        out.append(rec)
        print("%-16s %7.2f度 %8.3fm %8.4f  %-6s | %s%s"
              % (name, e["rot_deg"], e["trans"], e["scale_ratio"], rec["mode"],
                 ("%.2f度 %.3fm %.4f" % (s0["direct_rot_deg"], s0["direct_trans"],
                                         s0["direct_scale_ratio"])) if s0 else "—",
                 "   **成功**" if ok else ""), flush=True)
        # 1 件ごとに書き出す
        with open(args.out, "w") as f:
            json.dump({"provenance": provenance(), "seed": args.seed,
                       "results": out}, f, indent=2, ensure_ascii=False)

    n_ok = sum(r["success"] for r in out)
    print("\n**成功（回転<%.1f度・並進<%.2fm・縮尺比<%.2f）: %d / %d**"
          % (succ["rot_deg"], succ["trans"], succ["scale_ratio"], n_ok, len(out)))
    print("wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
