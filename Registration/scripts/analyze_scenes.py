"""Cross-scene analysis for the section-C multi-scene evaluation.

Reads per-scene benchmark outputs (results.csv / summary.json / trials.csv)
plus per-scene SLAM ATE (eval_ate.json) and produces:
  summary_table.csv / summary_table.md   scene x method aggregate table
  ate_vs_success.png                     ATE RMSE vs recovery success rate
  error_hist_<scene>.png                 per-trial error histograms + thresholds

Scene list comes from a YAML manifest (config-driven):
  scenes:
    - name: room_0
      bench_dir: output/Registration/sectionC/room_0
      ate_json: output/Replica/room0_official/260310_test4/eval_ate.json
  out_dir: output/Registration/sectionC_analysis

Usage:
  python Registration/scripts/analyze_scenes.py --manifest Registration/configs/sectionC/manifest.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

SERIES_1 = "#2a78d6"   # categorical slot 1 (existing thresholds)
SERIES_2 = "#1baf7a"   # categorical slot 2 (strict thresholds)
INK = "#333333"
GRID = dict(color="#cccccc", linewidth=0.6, alpha=0.6)
ERROR_PANELS = (("rot_deg", "rotation error [deg]"),
                ("trans", "translation error [m]"),
                ("scale_ratio", "scale error ratio"))


def _load_scene(entry: Dict) -> Optional[Dict]:
    bench = entry["bench_dir"]
    res_path = os.path.join(bench, "results.csv")
    if not os.path.isfile(res_path):
        print(f"  skip {entry['name']}: no {res_path}")
        return None
    with open(res_path) as f:
        rows = list(csv.DictReader(f))
    trials_path = os.path.join(bench, "trials.csv")
    trials = []
    if os.path.isfile(trials_path):
        with open(trials_path) as f:
            trials = list(csv.DictReader(f))
    summary = {}
    sum_path = os.path.join(bench, "summary.json")
    if os.path.isfile(sum_path):
        summary = json.load(open(sum_path))
    ate = None
    ate_json = entry.get("ate_json")
    if ate_json and os.path.isfile(ate_json):
        ate = json.load(open(ate_json))
    return {"name": entry["name"], "results": rows, "trials": trials,
            "summary": summary, "ate": ate}


def _fmt_dist(row: Dict, key: str, digits: int = 3) -> str:
    """median [q25, q75] (min-max) from a results.csv row."""
    med_key = {"rot_deg": "med_rot_deg", "trans": "med_trans",
               "scale_ratio": "med_scale_ratio"}[key]
    try:
        med = float(row[med_key])
        q25, q75 = float(row[f"{key}_q25"]), float(row[f"{key}_q75"])
        lo, hi = float(row[f"{key}_min"]), float(row[f"{key}_max"])
    except (KeyError, ValueError):
        return "-"
    d = digits
    return f"{med:.{d}f} [{q25:.{d}f}, {q75:.{d}f}] ({lo:.{d}f}-{hi:.{d}f})"


def _failure_str(summary: Dict, method: str, strict: bool = False) -> str:
    key = "failure_breakdown_strict" if strict else "failure_breakdown"
    fb = summary.get("methods", {}).get(method, {}).get(key)
    if not fb:
        return "-"
    combos = fb.get("combinations", {})
    if not combos:
        return "0 failed"
    parts = [f"{k}:{v}" for k, v in sorted(combos.items(), key=lambda kv: -kv[1])]
    return f"{fb['n_failed']} failed ({', '.join(parts)})"


def _write_tables(scenes: List[Dict], out_dir: str) -> List[Dict]:
    table: List[Dict] = []
    for sc in scenes:
        ate_rmse = None
        if sc["ate"]:
            ate_rmse = sc["ate"].get("absolute_translational_error.rmse")
        for row in sc["results"]:
            rec = {
                "scene": sc["name"],
                "method": row["method"],
                "trials": row["robust_trials"],
                "success_rate": row["success_rate"],
                "success_ci": f"[{row.get('success_ci_lo', '-')}, {row.get('success_ci_hi', '-')}]",
                "success_rate_strict": row.get("success_rate_strict", "-"),
                "success_strict_ci": f"[{row.get('success_strict_ci_lo', '-')}, {row.get('success_strict_ci_hi', '-')}]",
                "rot_deg (med [q25,q75] (min-max))": _fmt_dist(row, "rot_deg"),
                "trans_m": _fmt_dist(row, "trans", 4),
                "scale_ratio": _fmt_dist(row, "scale_ratio", 4),
                "failures": _failure_str(sc["summary"], row["method"]),
                "failures_strict": _failure_str(sc["summary"], row["method"], strict=True),
                "direct_chamfer_m": row["direct_chamfer"],
                "ate_rmse_cm": round(ate_rmse, 3) if ate_rmse is not None else "-",
            }
            table.append(rec)
    csv_path = os.path.join(out_dir, "summary_table.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        w.writeheader()
        w.writerows(table)
    md_path = os.path.join(out_dir, "summary_table.md")
    keys = list(table[0].keys())
    with open(md_path, "w") as f:
        f.write("| " + " | ".join(keys) + " |\n")
        f.write("|" + "|".join(["---"] * len(keys)) + "|\n")
        for rec in table:
            f.write("| " + " | ".join(str(rec[k]) for k in keys) + " |\n")
    print(f"wrote {csv_path} and {md_path}")
    return table


def _plot_ate_scatter(table: List[Dict], out_dir: str, method: str = "proposed") -> None:
    pts = [r for r in table
           if r["method"] == method and r["ate_rmse_cm"] != "-"]
    if not pts:
        print("no ATE data yet; skipping scatter")
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    x = [float(r["ate_rmse_cm"]) for r in pts]
    y = [float(r["success_rate"]) for r in pts]
    ax.scatter(x, y, s=70, color=SERIES_1, label="existing thresholds", zorder=3)
    ys = [r for r in pts if r["success_rate_strict"] not in ("-", "", None)]
    if ys:
        ax.scatter([float(r["ate_rmse_cm"]) for r in ys],
                   [float(r["success_rate_strict"]) for r in ys],
                   s=70, color=SERIES_2, marker="s",
                   label="strict thresholds", zorder=3)
    for r in pts:
        ax.annotate(r["scene"], (float(r["ate_rmse_cm"]), float(r["success_rate"])),
                    textcoords="offset points", xytext=(6, 4),
                    fontsize=8, color=INK)
    ax.set_xlabel("SLAM ATE RMSE [cm]", color=INK)
    ax.set_ylabel(f"recovery success rate ({method})", color=INK)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, **GRID)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    path = os.path.join(out_dir, "ate_vs_success.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


def _plot_histograms(sc: Dict, cfg_success: Dict, cfg_strict: Optional[Dict],
                     out_dir: str, method: str = "proposed") -> None:
    trials = [t for t in sc["trials"] if t["method"] == method]
    if not trials:
        return
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (key, label) in zip(axes, ERROR_PANELS):
        vals = np.array([float(t[key]) for t in trials])
        ax.hist(vals, bins=30, color=SERIES_1, edgecolor="white", linewidth=0.5)
        thr = cfg_success.get(key) if cfg_success else None
        if thr is not None:
            ax.axvline(thr, color=INK, linestyle="--", linewidth=1.2)
            ax.annotate(f"thr {thr}", (thr, ax.get_ylim()[1] * 0.95),
                        fontsize=8, color=INK, ha="left")
        if cfg_strict and cfg_strict.get(key) is not None:
            ax.axvline(cfg_strict[key], color="#888888", linestyle=":", linewidth=1.2)
            ax.annotate(f"strict {cfg_strict[key]}",
                        (cfg_strict[key], ax.get_ylim()[1] * 0.85),
                        fontsize=8, color="#888888", ha="left")
        ax.set_xlabel(label, color=INK)
        ax.grid(True, **GRID)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    axes[0].set_ylabel("trials", color=INK)
    fig.suptitle(f"{sc['name']} — per-trial recovery errors ({method}, "
                 f"n={len(trials)})", color=INK)
    fig.tight_layout()
    path = os.path.join(out_dir, f"error_hist_{sc['name']}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    args = ap.parse_args()
    manifest = yaml.safe_load(open(args.manifest))
    out_dir = manifest["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    scenes = [s for s in (_load_scene(e) for e in manifest["scenes"]) if s]
    if not scenes:
        raise SystemExit("no scene results found")
    table = _write_tables(scenes, out_dir)
    _plot_ate_scatter(table, out_dir)
    for sc in scenes:
        succ = sc["summary"].get("success_thresholds", {})
        strict = sc["summary"].get("success_thresholds_strict")
        _plot_histograms(sc, succ, strict, out_dir)


if __name__ == "__main__":
    main()
