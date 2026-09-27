"""R37 §4 の P 系列のうち、**摂動によって結果が変わった**試行を点群（PLY・PNG）に書き出す。**再計算なし。**

保存済みの $P$・$\\hat T_{on}$（`r37_replicate.json`）と、摂動なしの解（`r36_release.json` の `T_end`）を
source（seed 0）に当てるだけである。

置き方（すべて BIM 座標）
    摂動後・位置合わせ前（紫）: G1 · P · src
    位置合わせの結果 on（青）  : T_on · (P · src)
    摂動なしの解（水色）       : T_on(摂動なし) · src
    正解 G1（緑）              : G1 · src
    BIM（灰）

    conda activate sni-slam
    python Registration/scripts/r37_variant_export.py --out <出力先>
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys

import numpy as np
import yaml

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "Registration"))
sys.path.insert(0, os.path.join(REPO, "Registration", "scripts"))
os.chdir(REPO)

import matplotlib                                              # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                # noqa: E402

from criterion_verdict import d_omega                          # noqa: E402
from failure_decomposition import provenance                   # noqa: E402
from r37_visual import COL, G1, decimate, write_ply            # noqa: E402
from regbim import io_utils, metrics                           # noqa: E402

N = 30000
COL = dict(COL, pert=(0.60, 0.20, 0.75), base=(0.00, 0.75, 0.80))
# 摂動によって結果が変わった試行（R37 §4 の P 系列、on）。分類は 2026-09-27 の集計による
CASES = {
    "m3_cor_c__E2": ([3, 8], "向きの取り違え（180°）"),
    "m3_cor_b__E2": ([6], "向きの取り違え（90°）"),
    "m3_cor_c__E3": ([2, 5], "良い方に変わった（選ばれる候補が変わった）"),
    "m3_cor_b__E3": (list(range(10)), "向きは合ったまま、ばらつく（全 10 回）"),
    "m3_cor_d__E2": ([7], "少し良い方に変わった"),
    "m3_cor_d__E3": ([7], "少し良い方に変わった"),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rows = json.load(open("Registration/output/diag/r37_replicate.json"))["rows"]
    base = {r["target"]: np.asarray(r["T_end"], dtype=np.float64)
            for r in json.load(open("Registration/output/diag/r36_release.json"))["rows"]}
    omega = np.load("Registration/output/diag/omega.npz")
    man = {"provenance": provenance(), "n_points": N, "files": [], "cases": []}
    done = set()

    def ply(name, pts, key, target):
        idx = decimate(pts, N)
        write_ply(os.path.join(args.out, name), pts[idx], np.tile(COL[key], (len(idx), 1)))
        man["files"].append({"file": name, "target": target, "layer": key})

    for target, (ks, kind) in CASES.items():
        scene, cond = target.split("__")
        cfg = copy.deepcopy(yaml.safe_load(open("Registration/configs/realdata/%s.yaml" % target)))
        cfg["source"] = dict(cfg["source"], seed=0)
        src = io_utils.load_source_cloud(cfg)
        dst = io_utils.load_reference_cloud(cfg)
        G = G1(scene)
        idx = decimate(src.points, N)
        g1 = metrics.apply_sim3(G, src.points)
        b0 = metrics.apply_sim3(base[target], src.points)
        if ("bim", cond) not in done:
            ply("BIM_%s.ply" % cond, dst.points, "bim", target); done.add(("bim", cond))
        if ("g1", scene) not in done:
            ply("G1_%s.ply" % scene, g1, "g1", target); done.add(("g1", scene))
        ply("%s_nopert.ply" % target, b0, "base", target)
        for k in ks:
            r = next(x for x in rows if x["target"] == target and x["series"] == "P" and x["k"] == k)
            P = np.asarray(r["P"], dtype=np.float64)
            Ton = np.asarray(r["T_on"], dtype=np.float64)
            sp = metrics.apply_sim3(P, src.points)
            pert = metrics.apply_sim3(G, sp)
            on = metrics.apply_sim3(Ton, sp)
            ply("%s_P%d_pert.ply" % (target, k), pert, "pert", target)
            ply("%s_P%d_on.ply" % (target, k), on, "rel", target)

            fig, ax = plt.subplots(figsize=(12, 8), dpi=100)
            for pts, key in ((pert, "pert"), (g1, "g1"), (b0, "base"), (on, "rel")):
                q = pts[idx]
                ax.scatter(q[:, 0], q[:, 1], s=0.15, c=[COL[key]], linewidths=0, rasterized=True)
            bb = dst.points[decimate(dst.points, N)]
            ax.scatter(bb[:, 0], bb[:, 1], s=0.15, c=[COL["bim"]], linewidths=0, rasterized=True)
            ax.set_aspect("equal"); ax.set_xlabel("BIM X [m]"); ax.set_ylabel("BIM Y [m]")
            ax.set_title("%s P%d: BIM grey / G1 green / no-perturbation cyan / perturbed purple / result blue"
                         % (target, k))
            p = "%s_P%d_top.png" % (target, k)
            fig.savefig(os.path.join(args.out, p), bbox_inches="tight"); plt.close(fig)
            man["files"].append({"file": p, "target": target, "layer": "png"})

            Rp, tp, s = metrics.decompose_sim3(P)
            Tc = Ton @ P
            man["cases"].append({
                "target": target, "trial": "P%d" % k, "kind": kind,
                "perturb": {"rot_deg": float(np.degrees(np.arccos(np.clip((np.trace(Rp) - 1) / 2, -1, 1)))),
                            "trans_m": float(np.linalg.norm(tp)), "scale": float(s)},
                "gt_d_omega_m": r["on"]["decomposition_primary"]["d_omega_m"],
                "gt_rot_deg": r["on"]["primary"]["rot_deg"],
                "selfconsistency_d_omega_m": float(d_omega(Tc, base[target], omega[scene])),
                "selfconsistency_rot_deg": float(metrics.rotation_error_deg(
                    metrics.decompose_sim3(Tc)[0], metrics.decompose_sim3(base[target])[0])),
                "nopert_gt_d_omega_m": float(d_omega(base[target], G, omega[scene]))})
            print(target, "P%d" % k, flush=True)

    json.dump(man, open(os.path.join(args.out, "manifest.json"), "w"), indent=2, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
