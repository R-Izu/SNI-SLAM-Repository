"""R35 — 案A の 16 出力を 6 基準で採点し直し、$d_\\Omega$ を2項に分ける。

**手法は変えない。** R32 の 16 実行（4 シーン × E2/E3 × 既定・案A）を
R32 と同じ手順・同じ seed で再実行し、**最終変換 $\\hat T$ を保存する**
（R32 の `plan_a_compare.json` は $\\hat T$ を保存していなかった）。

段階
----
1. 実行（`--score-only` で省略）：`plan_a_compare.run` と同じ設定で 16 実行し、
   1 件ごとに $\\hat T$ を書き出す（途中で落ちても残す）
2. 採点：
   - R32 §1 の $d_\\Omega$ との差（R32 と同じ Ω = 各 target config から作る）
   - §3-1：6 基準（`criterion_spread.json`）× Ω（`omega.npz`。R30 §1-4 と同じ）
     で、回転・$d_\\Omega$・縮尺・AND。基準ごとに AND を取る
   - §3-2：主基準（基準1）に対する $d_\\Omega$ の2項分解

**閾値は変えない。** config の `eval.success` をそのまま使う。

    conda activate sni-slam
    python Registration/scripts/r35_rescore.py
    python Registration/scripts/r35_rescore.py --score-only
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from typing import Dict, List

import numpy as np
import yaml

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "Registration"))
sys.path.insert(0, os.path.join(REPO, "Registration", "scripts"))
os.chdir(REPO)

from criterion_verdict import build_omega, d_omega          # noqa: E402
from failure_decomposition import provenance                # noqa: E402
from regbim import io_utils, metrics                        # noqa: E402
from regbim.methods import get_method                       # noqa: E402

SCENES = ["m3_cor_a", "m3_cor_b", "m3_cor_c", "m3_cor_d"]
CONDS = ["E2", "E3"]
MODES = ["centroid", "plan_correlate"]
PRIMARY = 1            # R18 以来の主基準（現行 GT）


def register(cfg0: Dict, mode: str) -> Dict:
    """`plan_a_compare.run` と同じ設定で1回実行し、$\\hat T$ を返す。"""
    cfg = copy.deepcopy(cfg0)
    cfg["source"] = dict(cfg["source"], seed=0)
    cfg.setdefault("proposed", {})["translation_init"] = mode
    cfg.setdefault("diagnostics", {})
    cfg["diagnostics"]["record_yaw"] = True
    cfg["diagnostics"]["record_stages"] = True
    src = io_utils.load_source_cloud(cfg)
    dst = io_utils.load_reference_cloud(cfg)
    m = get_method("proposed")
    t0 = time.time()
    T = m.register(src, dst, cfg)
    return {"T_est": np.asarray(T, dtype=np.float64).tolist(),
            "time_register_s": round(time.time() - t0, 2)}


def decompose(T: np.ndarray, G: np.ndarray, omega: np.ndarray) -> Dict:
    """$d_\\Omega^2 = \\|A\\bar p+b\\|^2 + \\mathrm{tr}(A C_\\Omega A^\\top)$。

    $A=\\hat s\\hat R-s_GR_G$（= 両者の 3×3 ブロックの差）、$b=\\hat t-t_G$、
    $C_\\Omega$ は母共分散（1/N）。$Tp-Gp=Ap+b$ なので恒等式である。
    """
    A = T[:3, :3] - G[:3, :3]
    b = T[:3, 3] - G[:3, 3]
    pbar = omega.mean(axis=0)
    X = omega - pbar
    C = X.T @ X / len(omega)
    cen2 = float(np.sum((A @ pbar + b) ** 2))
    rs2 = float(np.trace(A @ C @ A.T))
    d = d_omega(T, G, omega)
    return {"centroid_m": float(np.sqrt(cen2)),
            "rot_scale_about_centroid_m": float(np.sqrt(rs2)),
            "d_omega_m": d,
            "abs_err_sq": abs(cen2 + rs2 - d * d),
            "rel_err_sq": abs(cen2 + rs2 - d * d) / (d * d) if d > 0 else None}


def score(rows: List[Dict], spread: Dict, omegas: Dict, r32: Dict) -> Dict:
    out = []
    for r in rows:
        scene, cond, mode = r["scene"], r["cond"], r["mode"]
        target = "%s__%s" % (scene, cond)
        cfg = yaml.safe_load(open("Registration/configs/realdata/%s.yaml" % target))
        th = cfg["eval"]["success"]
        T = np.asarray(r["T_est"], dtype=np.float64)
        R, _, s = metrics.decompose_sim3(T)

        # R32 と同じ Ω・同じ G で d_Ω を出し直し、R32 §1 の値と比べる
        omega32 = build_omega(dict(cfg, source=dict(cfg["source"], seed=0)))
        G32 = np.asarray(json.load(open(
            "output/GT_alignment_probe/T_gt/T_gt_%s.json" % scene))["T_gt"],
            dtype=np.float64).reshape(4, 4)
        d32_now = d_omega(T, G32, omega32)
        d32_then = r32.get((target, mode))

        # §3-1：6 基準
        per = []
        for f in spread[scene]["results"]:
            if "T" not in f:
                continue
            G = np.asarray(f["T"], dtype=np.float64)
            RG, _, sG = metrics.decompose_sim3(G)
            e = {"id": f["id"],
                 "rot_deg": metrics.rotation_error_deg(R, RG),
                 "d_omega": d_omega(T, G, omegas[scene]),
                 "scale_ratio": float(abs(s / sG - 1.0))}
            e["ok"] = bool(e["rot_deg"] < th["rot_deg"]
                           and e["d_omega"] < th["trans"]
                           and e["scale_ratio"] < th["scale_ratio"])
            per.append(e)
        n_ok = sum(p["ok"] for p in per)
        verdict = ("all_pass" if n_ok == len(per) else
                   "all_fail" if n_ok == 0 else "criterion_dependent")
        prim = next(p for p in per if p["id"] == PRIMARY)
        Gp = np.asarray(next(f["T"] for f in spread[scene]["results"]
                             if f["id"] == PRIMARY), dtype=np.float64)
        degen = bool(metrics.sim3_errors(T, Gp)["degenerate"])

        out.append({
            "target": target, "mode": mode,
            "T_est": r["T_est"],
            "r32_d_omega_reported": d32_then,
            "r32_d_omega_rerun": d32_now,
            "r32_diff": (None if d32_then is None else abs(d32_now - d32_then)),
            "omega_cache_vs_target_cfg_maxabs": float(
                np.abs(omegas[scene] - omega32).max())
            if omegas[scene].shape == omega32.shape else "shape_mismatch",
            "per_criterion": per, "n_ok": n_ok, "n_criteria": len(per),
            "verdict": verdict,
            "primary": dict(prim, degenerate=degen,
                            success=bool(prim["ok"] and not degen)),
            "decomposition_primary": decompose(T, Gp, omegas[scene]),
            "thresholds": th,
        })
    return {"rows": out}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="Registration/output/diag/r35_runs.json")
    ap.add_argument("--r32", default="Registration/output/diag/plan_a_compare.json")
    ap.add_argument("--spread", default="Registration/output/diag/criterion_spread.json")
    ap.add_argument("--omega-cache", default="Registration/output/diag/omega.npz")
    ap.add_argument("--out", default="Registration/output/diag/r35_rescore.json")
    ap.add_argument("--score-only", action="store_true")
    args = ap.parse_args()

    # 1. 実行（1 件ごとに書き出す）
    if not args.score_only:
        rows: List[Dict] = []
        for scene in SCENES:
            for cond in CONDS:
                target = "%s__%s" % (scene, cond)
                cfg0 = yaml.safe_load(open("Registration/configs/realdata/%s.yaml" % target))
                for mode in MODES:
                    r = dict(scene=scene, cond=cond, mode=mode, **register(cfg0, mode))
                    rows.append(r)
                    print("%-16s %-16s %.1f s" % (target, mode, r["time_register_s"]),
                          flush=True)
                    with open(args.runs, "w") as f:
                        json.dump({"provenance": provenance(), "rows": rows}, f, indent=2)

    runs = json.load(open(args.runs))
    r32j = json.load(open(args.r32))
    r32 = {(x["target"], x["mode"]): x.get("final_d_omega_m") for x in r32j["rows"]}
    spread = json.load(open(args.spread))["scenes"]
    z = np.load(args.omega_cache)
    omegas = {k: z[k] for k in z.files}

    res = score(runs["rows"], spread, omegas, r32)
    res.update({"provenance_score": provenance(),
                "provenance_runs": runs["provenance"],
                "provenance_r32": r32j["provenance"],
                "primary_criterion": PRIMARY})
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

    # 表示
    print("\n## R32 §1 との差")
    for x in res["rows"]:
        print("%-16s %-16s R32 %.6f  今回 %.6f  差 %.2e  Ω差 %s"
              % (x["target"], x["mode"], x["r32_d_omega_reported"] or float("nan"),
                 x["r32_d_omega_rerun"], x["r32_diff"] or 0.0,
                 x["omega_cache_vs_target_cfg_maxabs"]))
    print("  差の最大 = %.3e m"
          % max(x["r32_diff"] for x in res["rows"] if x["r32_diff"] is not None))
    print("\n## §3-1 6 基準")
    for x in res["rows"]:
        print("%-16s %-16s %s %d/6 主=%s"
              % (x["target"], x["mode"], x["verdict"], x["n_ok"],
                 "○" if x["primary"]["success"] else "×"))
        for p in x["per_criterion"]:
            print("    G%d  回転 %7.3f°  dΩ %8.4f m  縮尺 %.4f  %s"
                  % (p["id"], p["rot_deg"], p["d_omega"], p["scale_ratio"],
                     "○" if p["ok"] else "×"))
    print("\n## §3-2 分解（主基準）")
    for x in res["rows"]:
        d = x["decomposition_primary"]
        print("%-16s %-16s 重心 %.4f  回転縮尺 %.4f  dΩ %.4f  縮尺 %.4f  回転 %.3f°  誤差 %.1e"
              % (x["target"], x["mode"], d["centroid_m"],
                 d["rot_scale_about_centroid_m"], d["d_omega_m"],
                 x["primary"]["scale_ratio"], x["primary"]["rot_deg"], d["abs_err_sq"]))
    print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
