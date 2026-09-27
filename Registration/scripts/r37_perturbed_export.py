"""R37 §4 の P 系列から、摂動 P0 の試行を 8 条件ぶん点群（PLY・PNG）に書き出す。**再計算なし。**

`r37_replicate.json` に保存した $P$・$\\hat T_{off}$・$\\hat T_{on}$ を、source（seed 0）に当てるだけである。

置き方（すべて BIM 座標）
    摂動後・位置合わせ前（紫）: G1 · P · src      … 正解の置き方に、摂動 P をかけた状態
    off の解（赤）            : T_off · (P · src)
    on の解（青）             : T_on  · (P · src)
    正解 G1（緑）             : G1 · src
    BIM（灰）

    conda activate sni-slam
    python Registration/scripts/r37_perturbed_export.py --out <出力先>
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

COL = dict(COL, pert=(0.60, 0.20, 0.75))                       # 紫：摂動後・位置合わせ前
K = 0                                                          # P0 に固定（結果を見て選ばない）


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rows = [r for r in json.load(open("Registration/output/diag/r37_replicate.json"))["rows"]
            if r["series"] == "P" and r["k"] == K]
    omega = np.load("Registration/output/diag/omega.npz")
    man = {"provenance": provenance(), "trial": "P%d" % K, "files": [], "conditions": []}
    done_bim, done_g1 = set(), set()

    for r in rows:
        target = r["target"]
        scene, cond = target.split("__")
        cfg = yaml.safe_load(open("Registration/configs/realdata/%s.yaml" % target))
        cfg = copy.deepcopy(cfg)
        cfg["source"] = dict(cfg["source"], seed=0)
        src = io_utils.load_source_cloud(cfg)
        dst = io_utils.load_reference_cloud(cfg)
        P = np.asarray(r["P"], dtype=np.float64)
        Toff = np.asarray(r["T_off"], dtype=np.float64)
        Ton = np.asarray(r["T_on"], dtype=np.float64)
        G = G1(scene)
        sp = metrics.apply_sim3(P, src.points)              # 摂動を加えた source（source 座標）
        lay = {"pert": metrics.apply_sim3(G, sp), "off": metrics.apply_sim3(Toff, sp),
               "on": metrics.apply_sim3(Ton, sp), "g1": metrics.apply_sim3(G, src.points)}
        idx = decimate(src.points)

        def ply(name, pts, key):
            write_ply(os.path.join(args.out, name), pts[idx] if len(pts) == len(src.points)
                      else pts[decimate(pts)], np.tile(COL[key], (min(len(pts), 50000), 1)))
            man["files"].append({"file": name, "target": target, "layer": key})

        if cond not in done_bim:
            ply("BIM_%s.ply" % cond, dst.points, "bim"); done_bim.add(cond)
        if scene not in done_g1:
            ply("G1_%s.ply" % scene, lay["g1"], "g1"); done_g1.add(scene)
        for key, col in (("pert", "pert"), ("off", "r35"), ("on", "rel")):
            ply("%s_P%d_%s.ply" % (target, K, key), lay[key], col)

        # PNG（真上）：灰・緑・紫・青。赤は青とほぼ重なるので PLY だけにする
        fig, ax = plt.subplots(figsize=(12, 8), dpi=110)
        for key, col in (("pert", "pert"), ("g1", "g1"), ("on", "rel")):
            q = lay[key][idx]
            ax.scatter(q[:, 0], q[:, 1], s=0.15, c=[COL[col]], linewidths=0, rasterized=True)
        b = dst.points[decimate(dst.points)]
        ax.scatter(b[:, 0], b[:, 1], s=0.15, c=[COL["bim"]], linewidths=0, rasterized=True)
        ax.set_aspect("equal"); ax.set_xlabel("BIM X [m]"); ax.set_ylabel("BIM Y [m]")
        ax.set_title("%s P%d top: BIM grey / G1 green / perturbed (before) purple / result (on) blue"
                     % (target, K))
        p = "%s_P%d_top.png" % (target, K)
        fig.savefig(os.path.join(args.out, p), bbox_inches="tight"); plt.close(fig)
        man["files"].append({"file": p, "target": target, "layer": "png"})

        Rp, tp, spc = metrics.decompose_sim3(P)
        man["conditions"].append({
            "target": target,
            "perturb": {"rot_deg": float(np.degrees(np.arccos(np.clip((np.trace(Rp) - 1) / 2, -1, 1)))),
                        "trans_m": float(np.linalg.norm(tp)), "scale": float(spc)},
            "d_omega_before_m": float(d_omega(G @ P, G, omega[scene])),
            "off": {"d_omega_m": r["off"]["decomposition_primary"]["d_omega_m"],
                    "rot_deg": r["off"]["primary"]["rot_deg"], "scale_ratio": r["off"]["primary"]["scale_ratio"]},
            "on": {"d_omega_m": r["on"]["decomposition_primary"]["d_omega_m"],
                   "rot_deg": r["on"]["primary"]["rot_deg"], "scale_ratio": r["on"]["primary"]["scale_ratio"],
                   "success": r["on"]["primary"]["success"]}})
        print(target, "done", flush=True)

    json.dump(man, open(os.path.join(args.out, "manifest.json"), "w"), indent=2, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
