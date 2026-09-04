"""実データの位置合わせを全条件で回す（10 シーン × E1〜E4）。

**1シーンも母数から外さない**（追補1 Q7 / [[PROF]] C11）。
`m3_block_d` のように入力が壊れているものも回し、層別で報告する。

事前登録した予測（[[2026-08-30_preregistration_yaw_hypothesis]]）を検証する：
  P1 壁方向が1方向の廊下4本は、2方向の室6本より成功率が低い
  P2 失敗した試行ではヨー候補の取り違えが過半を占める
  P3 取り違えたときの候補スコアの差（margin）は、正解したときより小さい

    conda activate sni-slam
    python Registration/scripts/run_realdata.py --trials 100 --jobs 8
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from typing import Dict, List

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", default="Registration/configs/realdata")
    ap.add_argument("--out-root", default="Registration/output/realdata")
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--only", nargs="*", default=None, help="条件を絞る 例 E1 E2")
    args = ap.parse_args()
    repo = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "..", ".."))
    os.chdir(repo)

    cfgs = sorted(glob.glob(os.path.join(args.configs, "*.yaml")))
    if args.only:
        cfgs = [c for c in cfgs
                if os.path.basename(c).rsplit("__", 1)[1][:-5] in args.only]
    os.makedirs(args.out_root, exist_ok=True)
    print("%d 実行（%d 並列・%d 試行）" % (len(cfgs), args.jobs, args.trials))

    env = dict(os.environ)
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS"):
        env[k] = str(args.threads)

    pending, running, done, t0 = list(cfgs), [], 0, time.time()
    while pending or running:
        while pending and len(running) < args.jobs:
            c = pending.pop(0)
            key = os.path.basename(c)[:-5]
            out = os.path.join(args.out_root, key)
            if os.path.exists(os.path.join(out, "summary.json")):
                done += 1
                continue                       # 再開できるようにする
            os.makedirs(out, exist_ok=True)
            log = open(os.path.join(out, "run.log"), "w")
            p = subprocess.Popen(
                [sys.executable, "-W", "ignore", "Registration/scripts/benchmark.py",
                 "--config", c, "--methods", "proposed",
                 "--trials", str(args.trials), "--out-dir", out],
                stdout=log, stderr=subprocess.STDOUT, env=env)
            running.append((p, key, log))
        time.sleep(5)
        for item in list(running):
            p, key, log = item
            if p.poll() is None:
                continue
            running.remove(item)
            log.close()
            done += 1
            print("[%6.1f 分] %3d/%3d %-24s rc=%d"
                  % ((time.time() - t0) / 60, done, len(cfgs), key, p.returncode),
                  flush=True)

    # --- 収集 ---
    rows: List[Dict] = []
    for c in cfgs:
        key = os.path.basename(c)[:-5]
        scene, cond = key.rsplit("__", 1)
        out = os.path.join(args.out_root, key)
        sp = os.path.join(out, "summary.json")
        if not os.path.exists(sp):
            rows.append({"scene": scene, "condition": cond, "status": "failed"})
            continue
        agg = json.load(open(sp))["methods"]["proposed"]["aggregate"]
        fb = json.load(open(sp))["methods"]["proposed"].get("failure_breakdown")
        yaw = None
        yp = os.path.join(out, "yaw_diag.json")
        if os.path.exists(yp):
            recs = json.load(open(yp))
            if recs:
                wrong = [r for r in recs if not r["picked_correct"]]
                mw = [r["margin"] for r in wrong if r.get("margin") is not None]
                mr = [r["margin"] for r in recs
                      if r["picked_correct"] and r.get("margin") is not None]
                yaw = {
                    "n": len(recs),
                    "frac_picked_correct": round(
                        sum(r["picked_correct"] for r in recs) / len(recs), 4),
                    "median_margin_when_wrong": (round(float(np.median(mw)), 5)
                                                 if mw else None),
                    "median_margin_when_right": (round(float(np.median(mr)), 5)
                                                 if mr else None),
                    "frac_success_given_correct_yaw": (round(float(np.mean(
                        [r["success"] for r in recs if r["picked_correct"]])), 4)
                        if any(r["picked_correct"] for r in recs) else None),
                    "frac_success_given_wrong_yaw": (round(float(np.mean(
                        [r["success"] for r in wrong])), 4) if wrong else None),
                }
        rows.append({"scene": scene, "condition": cond, "status": "ok",
                     "success_rate": agg["success_rate"],
                     "ci_lo": agg.get("success_ci_lo"), "ci_hi": agg.get("success_ci_hi"),
                     "trials": agg["robust_trials"],
                     "direct_rot_deg": agg["direct_rot_deg"],
                     "direct_trans": agg["direct_trans"],
                     "direct_scale_ratio": agg["direct_scale_ratio"],
                     "direct_chamfer": agg["direct_chamfer"],
                     "direct_inlier": agg["direct_inlier"],
                     "med_rot_deg": agg["med_rot_deg"], "med_trans": agg["med_trans"],
                     "med_scale_ratio": agg["med_scale_ratio"],
                     "failure_breakdown": fb, "yaw": yaw})
    with open(os.path.join(args.out_root, "realdata_results.json"), "w") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print("\nwrote %s" % os.path.join(args.out_root, "realdata_results.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
