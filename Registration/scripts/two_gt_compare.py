"""R21 §3-1 — 旧 GT と新 GT の両方で測り、両方を報告する。

**どちらを GT にするかを選ばない。** 選べば、その判断が全数値に乗る。
両方出せば、**「正解基準の作り方が結果をどれだけ動かすか」**が読者に見える。

出すもの（旧／新／差の3列）：

1. 実データ 40 条件の GT 基準の誤差（回転・$d_\\Omega$・縮尺比）と成功数
2. 双方向採点の $k$（(iii) の実行で $\\rho_Q$ が正しい候補を1位にする数）

**手法は再実行しない。** 保存済みの `T_est`（40 条件）と `yaw_candidate_T`（960 候補）を
採点し直すだけである（R7 §3-2）。

**$d_\\Omega$ は R17 §5 の固定評価点集合で測る**（並進ベクトル差でも重心でもない）。
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys
from typing import Dict, List

import numpy as np
import yaml

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "Registration"))
sys.path.insert(0, os.path.join(REPO, "Registration", "scripts"))
os.chdir(REPO)

from criterion_verdict import d_omega                        # noqa: E402
from failure_decomposition import classify, provenance, wilson   # noqa: E402
from regbim.metrics import decompose_sim3, rotation_error_deg    # noqa: E402

KITS = {"old": "output/GT_alignment", "new": "output/GT_alignment_probe"}


def gt_of(kit: str, scene: str) -> np.ndarray:
    p = os.path.join(kit, "T_gt", "T_gt_%s.json" % scene)
    return np.asarray(json.load(open(p))["T_gt"], dtype=np.float64).reshape(4, 4)


def part1(omegas: Dict[str, np.ndarray]) -> List[Dict]:
    """40 条件の誤差を両 GT で。"""
    recs = json.load(open("Registration/output/diag/realdata_direct_v3.json"))["results"]
    out = []
    print("=== 1. 実データ 40 条件（無摂動解、seed 0 固定）===")
    print("%-16s %-24s %-24s" % ("target", "旧 GT  回転/d_Ω/縮尺", "新 GT  回転/d_Ω/縮尺"))
    n_ok = {"old": 0, "new": 0}
    for r in sorted(recs, key=lambda x: x["target"]):
        T = np.asarray(r["T_est"], dtype=np.float64)
        cfg = yaml.safe_load(open("Registration/configs/realdata/%s.yaml" % r["target"]))
        th = cfg["eval"]["success"]
        line = "%-16s" % r["target"]
        rec = {"target": r["target"], "scene": r["scene"]}
        for tag, kit in KITS.items():
            G = gt_of(kit, r["scene"])
            e = {"rot_deg": rotation_error_deg(decompose_sim3(T)[0], decompose_sim3(G)[0]),
                 "d_omega": d_omega(T, G, omegas[r["scene"]]),
                 "scale_ratio": float(abs(decompose_sim3(T)[2] / decompose_sim3(G)[2] - 1.0))}
            e["success"] = bool(e["rot_deg"] < th["rot_deg"] and e["d_omega"] < th["trans"]
                                and e["scale_ratio"] < th["scale_ratio"])
            n_ok[tag] += e["success"]
            rec[tag] = e
            line += " %7.2f/%7.3f/%6.4f%s" % (e["rot_deg"], e["d_omega"],
                                              e["scale_ratio"], "*" if e["success"] else " ")
        out.append(rec)
        print(line)
    print("\n  **成功（回転<%.1f度 ∧ d_Ω<%.2fm ∧ 縮尺比<%.2f）: 旧 %d/40 / 新 %d/40**"
          % (th["rot_deg"], th["trans"], th["scale_ratio"], n_ok["old"], n_ok["new"]))
    return out


def part2(rot_tol: float, basin: float, omegas) -> Dict:
    """双方向採点の k を両 GT で。"""
    scores = json.load(open("Registration/output/diag/bidirectional_scores.json"))["records"]
    rho = {(r["sweep"], r["mode"], r["seed"], r["cand"]): r for r in scores}
    sweeps = sorted(glob.glob("Registration/output/diag/sweep_*_io[01].json"))
    out = {}
    print("\n=== 2. 双方向採点：(iii) の件数と k（ρ_Q 単独で正しい候補が1位）===")
    for tag, kit in KITS.items():
        n_iii = k = 0
        for path in sweeps:
            blob = json.load(open(path))
            cfg = yaml.safe_load(open(blob["config"]))
            scene = os.path.basename(blob["config"])[:-5].split("__")[0]
            G = gt_of(kit, scene)
            from regbim import io_utils
            c2 = dict(cfg, reference=dict(cfg["reference"],
                                          inner_only=bool(blob.get("inner_only"))))
            p0 = io_utils.load_source_cloud(c2).points.mean(axis=0)
            for row in blob.get("rows") or []:
                cls = classify(row, G, rot_tol, basin, p0)
                if cls["class"] != "iii_lost_on_score":
                    continue
                n_iii += 1
                cands = [rho.get((os.path.basename(path), row["mode"], row["seed"], i))
                         for i in range(len(row["yaw_candidate_T"]))]
                if any(c is None for c in cands):
                    continue
                # 「正しい候補」は GT ごとに定義し直す
                gp = None
                best = cls["best_near_idx"]
                w = int(np.argmax([c["rho_Q"] for c in cands]))
                k += int(w == best)
        lo, hi = wilson(k, n_iii) if n_iii else (float("nan"),) * 2
        out[tag] = {"n_iii": n_iii, "k": k, "ci": [lo, hi]}
        print("  %-4s GT : (iii) %3d 件 / k = %d  (%.3f) [%.3f, %.3f]"
              % (tag, n_iii, k, k / max(n_iii, 1), lo, hi))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rot-tol", type=float, default=5.0)
    ap.add_argument("--basin", type=float, default=1.0)
    ap.add_argument("--out", default="Registration/output/diag/two_gt_compare.json")
    args = ap.parse_args()

    z = np.load("Registration/output/diag/omega.npz")
    omegas = {k: z[k] for k in z.files}

    p1 = part1(omegas)
    p2 = part2(args.rot_tol, args.basin, omegas)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"provenance": provenance(), "conditions": p1, "bidirectional": p2},
                  f, indent=2, ensure_ascii=False)
    print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
