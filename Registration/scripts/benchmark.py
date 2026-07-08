"""Benchmark every registered method on SLAM -> reference and rank them.

Two parts per method:
  * direct: register once, report chamfer / semantic inlier ratio / time (and
    error vs T_gt if frozen).
  * robustness: apply N known Sim3 perturbations to the source and check the
    method recovers the alignment (rotation/translation/scale within thresholds).

Writes results.csv + comparison figures and records the best method (highest
recovery success rate, ties broken by chamfer) to adopted.json. Experimental
methods dropped in regbim/methods/experimental/ are included automatically.
"""

import argparse
import json
import os
import subprocess
import time
from typing import Dict, List, Optional, Tuple

import _bootstrap  # noqa: F401
import numpy as np

from regbim import io_utils, metrics, stats
from regbim.config import load_config, load_t_gt
from regbim.methods import available_methods, get_method


def _evaluate(method_name: str, src, dst, T_ref_or_none: Optional[np.ndarray],
              cfg: Dict, trials: int, seed: int) -> Tuple[Dict, List[Dict]]:
    method = get_method(method_name)
    thr = cfg["semantic_icp"]["max_corr_dist"]
    succ = cfg["eval"]["success"]
    succ_strict = cfg["eval"].get("success_strict")
    # Fresh generator per method: every method faces the identical
    # perturbation sequence (paired comparison, fully determined by the seed).
    rng = np.random.default_rng(seed)

    t0 = time.time()
    T0 = method.register(src, dst, cfg)
    direct_time = time.time() - t0
    direct_chamfer = metrics.chamfer_distance(metrics.apply_sim3(T0, src.points), dst.points)
    direct_inlier = metrics.class_inlier_ratio(src, dst, T0, thr)
    direct_err = metrics.sim3_errors(T0, T_ref_or_none) if T_ref_or_none is not None else None

    # Robustness = self-consistency: does the method return to ITS OWN unperturbed
    # alignment T0 after a known Sim3 perturbation? (Fair across methods; T_gt is
    # only used for the direct error metric above.)
    ref_T = T0
    trial_records: List[Dict] = []
    rot_errs, trans_errs, scale_errs, times, successes = [], [], [], [], []
    successes_strict = []
    for i in range(trials):
        P = metrics.random_sim3(rng, cfg["eval"]["perturb"])
        src_p = metrics.transform_cloud(src, P)
        ts = time.time()
        Tt = method.register(src_p, dst, cfg)
        dt = time.time() - ts
        times.append(dt)
        expected = ref_T @ metrics.invert_sim3(P)
        e = metrics.sim3_errors(Tt, expected)
        rot_errs.append(e["rot_deg"]); trans_errs.append(e["trans"]); scale_errs.append(e["scale_ratio"])
        ok = stats.check_success(e, succ)
        successes.append(bool(ok))
        ok_strict = stats.check_success(e, succ_strict) if succ_strict else None
        if ok_strict is not None:
            successes_strict.append(bool(ok_strict))
        Rp, tp, sp = metrics.decompose_sim3(P)
        trial_records.append({
            "method": method_name,
            "trial": i,
            "pert_rot_deg": round(metrics.rotation_error_deg(Rp, np.eye(3)), 3),
            "pert_trans": round(float(np.linalg.norm(tp)), 4),
            "pert_scale": round(float(sp), 4),
            "rot_deg": round(e["rot_deg"], 4),
            "trans": round(e["trans"], 5),
            "scale_ratio": round(e["scale_ratio"], 5),
            "time_s": round(dt, 3),
            "success": bool(ok),
            "success_strict": ok_strict,
        })

    def med(x):
        return float(np.median(x)) if x else float("nan")

    row = {
        "method": method_name,
        "direct_time_s": round(direct_time, 3),
        "direct_chamfer": round(direct_chamfer, 4),
        "direct_inlier": round(direct_inlier, 3),
        "direct_rot_deg": None if direct_err is None else round(direct_err["rot_deg"], 3),
        "direct_trans": None if direct_err is None else round(direct_err["trans"], 4),
        "direct_scale_ratio": None if direct_err is None else round(direct_err["scale_ratio"], 4),
        "robust_trials": trials,
        "success_rate": round(float(np.mean(successes)), 3) if successes else float("nan"),
        "med_rot_deg": round(med(rot_errs), 3),
        "med_trans": round(med(trans_errs), 4),
        "med_scale_ratio": round(med(scale_errs), 4),
        "med_time_s": round(med(times), 3),
    }

    # Appended columns (existing columns above keep their order for
    # backward compatibility with older results.csv consumers).
    ci_lo, ci_hi = stats.wilson_ci(int(np.sum(successes)), len(successes))
    row["success_ci_lo"] = round(ci_lo, 3)
    row["success_ci_hi"] = round(ci_hi, 3)
    if successes_strict:
        s_lo, s_hi = stats.wilson_ci(int(np.sum(successes_strict)), len(successes_strict))
        row["success_rate_strict"] = round(float(np.mean(successes_strict)), 3)
        row["success_strict_ci_lo"] = round(s_lo, 3)
        row["success_strict_ci_hi"] = round(s_hi, 3)
    else:
        row["success_rate_strict"] = None
        row["success_strict_ci_lo"] = None
        row["success_strict_ci_hi"] = None
    for key, errs in (("rot_deg", rot_errs), ("trans", trans_errs), ("scale_ratio", scale_errs)):
        pct = stats.error_percentiles(errs)
        for stat_name in ("min", "q25", "q75", "max", "mean", "std"):
            row[f"{key}_{stat_name}"] = round(pct[stat_name], 5)
    return row, trial_records


def _write_csv(rows: List[Dict], path: str) -> None:
    import csv
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _git_commit() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            text=True).strip()
    except Exception:
        return None


def _write_summary(rows: List[Dict], all_trials: List[Dict], cfg: Dict,
                   config_path: str, trials: int, out_dir: str) -> None:
    """summary.json: reproducibility metadata + per-method failure breakdown."""
    succ = cfg["eval"]["success"]
    succ_strict = cfg["eval"].get("success_strict")
    per_method = {}
    for row in rows:
        name = row["method"]
        errs = [{"rot_deg": t["rot_deg"], "trans": t["trans"],
                 "scale_ratio": t["scale_ratio"]}
                for t in all_trials if t["method"] == name]
        entry = {"aggregate": row,
                 "failure_breakdown": stats.failure_breakdown(errs, succ)}
        if succ_strict:
            entry["failure_breakdown_strict"] = stats.failure_breakdown(errs, succ_strict)
        per_method[name] = entry
    summary = {
        "config": os.path.abspath(config_path),
        "seed": int(cfg["eval"]["seed"]),
        "trials": trials,
        "perturb": cfg["eval"]["perturb"],
        "success_thresholds": succ,
        "success_thresholds_strict": succ_strict,
        "git_commit": _git_commit(),
        "rng_scheme": "per-method reseed (identical perturbation sequence per method)",
        "methods": per_method,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


def _write_figures(rows: List[Dict], out_dir: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [r["method"] for r in rows]
    panels = [("success_rate", "Recovery success rate"),
              ("direct_chamfer", "Direct chamfer (m)"),
              ("med_rot_deg", "Median rotation error (deg)"),
              ("med_time_s", "Median time / run (s)")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (key, title) in zip(axes.ravel(), panels):
        vals = [r[key] if r[key] is not None else 0 for r in rows]
        ax.bar(names, vals, color="#4C78A8")
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "comparison.png"), dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--trials", type=int, default=None)
    ap.add_argument("--methods", nargs="*", default=None, help="subset to run")
    args = ap.parse_args()
    cfg = load_config(args.config)
    trials = args.trials if args.trials is not None else int(cfg["eval"]["trials"])
    seed = int(cfg["eval"]["seed"])

    src = io_utils.load_source_cloud(cfg)
    dst = io_utils.load_reference_cloud(cfg)
    T_gt = load_t_gt(cfg)
    if T_gt is None:
        print("WARNING: no frozen T_gt; recovery is measured relative to each "
              "method's own direct result. Run establish_gt.py for absolute error.")

    methods = args.methods or available_methods()
    print(f"benchmarking methods: {methods}  (trials={trials}, seed={seed})")
    rows = []
    all_trials: List[Dict] = []
    for name in methods:
        print(f"  -> {name}")
        row, trial_records = _evaluate(name, src, dst, T_gt, cfg, trials, seed)
        rows.append(row)
        all_trials.extend(trial_records)

    out_dir = cfg["eval"]["out_dir"]
    _write_csv(rows, os.path.join(out_dir, "results.csv"))
    _write_csv(all_trials, os.path.join(out_dir, "trials.csv"))
    _write_summary(rows, all_trials, cfg, args.config, trials, out_dir)
    _write_figures(rows, out_dir)

    # Best: highest success rate, ties broken by lowest direct chamfer.
    best = sorted(rows, key=lambda r: (-r["success_rate"], r["direct_chamfer"]))[0]
    with open(os.path.join(out_dir, "adopted.json"), "w") as f:
        json.dump({"adopted_method": best["method"], "ranking": rows}, f, indent=2)

    print("\n=== results ===")
    for r in rows:
        print(f"{r['method']:18s} success={r['success_rate']}  chamfer={r['direct_chamfer']}  "
              f"med_rot={r['med_rot_deg']}deg  med_time={r['med_time_s']}s")
    print(f"\nADOPTED: {best['method']}  (results in {out_dir}/)")


if __name__ == "__main__":
    main()
