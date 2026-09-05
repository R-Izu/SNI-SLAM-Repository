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
    # Rigid methods can't estimate scale: pre-apply the GT (perturbation) scale
    # as a fairness handicap. Declared via the method's ``gt_scale`` attribute;
    # recorded per-row so summary.json documents the protocol.
    gt_scale = bool(getattr(method, "gt_scale", False))

    t0 = time.time()
    T0 = method.register(src, dst, cfg)
    direct_time = time.time() - t0
    direct_chamfer = metrics.chamfer_distance(metrics.apply_sim3(T0, src.points), dst.points)
    direct_inlier = metrics.class_inlier_ratio(src, dst, T0, thr)
    direct_err = metrics.sim3_errors(T0, T_ref_or_none) if T_ref_or_none is not None else None

    # Two different questions, measured separately (R7 §3-1).
    #
    #   self-consistency : does the method return to ITS OWN unperturbed alignment
    #                      T0 after a known Sim3 perturbation? Fair across methods,
    #                      needs no ground truth -- but a method that always lands
    #                      on the SAME WRONG pose scores perfectly.
    #   gt               : does it land on the reference transform T_gt? This is
    #                      accuracy. It only exists when T_gt is independent of the
    #                      method being evaluated -- check the provenance before
    #                      reading it (Replica's frozen T_gt is `proposed`'s own
    #                      output, so `gt_*` is meaningless there).
    #
    # These were previously conflated under the name `success_rate`, which is how a
    # self-consistency number ended up being reported as an accuracy. Both are kept,
    # and the ambiguous name is gone.
    ref_T = T0
    trial_records: List[Dict] = []
    yaw_records: List[Dict] = []
    # trial = -1 は無摂動の解 T0。自己一貫性の基準そのものなので必ず残す。
    matrix_records: List[Dict] = [
        {"method": method_name, "trial": -1,
         "T_est": np.asarray(T0, dtype=np.float64).tolist(), "P": None}]
    rot_errs, trans_errs, scale_errs, times, successes = [], [], [], [], []
    gt_rot, gt_trans, gt_scale_e, gt_successes = [], [], [], []
    successes_strict = []
    for i in range(trials):
        P = metrics.random_sim3(rng, cfg["eval"]["perturb"])
        if gt_scale:
            # Same draw (keeps the perturbation sequence paired across methods),
            # then undo the scale component = "GT scale pre-applied" protocol.
            Rp_, tp_, sp_ = metrics.decompose_sim3(P)
            P = metrics.make_sim3(Rp_, tp_ / sp_, 1.0)
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
        # R7 §3-1: the same trial, scored against the reference transform instead
        # of against the method's own answer.
        e_gt = ok_gt = None
        if T_ref_or_none is not None:
            expected_gt = T_ref_or_none @ metrics.invert_sim3(P)
            e_gt = metrics.sim3_errors(Tt, expected_gt)
            ok_gt = bool(stats.check_success(e_gt, succ))
            gt_rot.append(e_gt["rot_deg"]); gt_trans.append(e_gt["trans"])
            gt_scale_e.append(e_gt["scale_ratio"]); gt_successes.append(ok_gt)
        Rp, tp, sp = metrics.decompose_sim3(P)
        # R5 §2-5: which Manhattan yaw candidate was picked, and was it the right
        # one? The correct candidate can only be decided here, because only the
        # evaluator knows the expected rotation. Opt-in via config, so every
        # existing config keeps producing byte-identical output.
        yaw_diag = None
        if (cfg.get("diagnostics") or {}).get("record_yaw"):
            d = getattr(method, "last_yaw_diag", None)
            if d:
                R_exp = metrics.decompose_sim3(expected)[0]
                errs = [metrics.rotation_error_deg(np.asarray(R), R_exp)
                        for R in d["candidate_R"]]
                w = d["winner"]
                # ★ 命名規約（R7 §3-4）：自己一貫性の量は必ず `selfconsistency_` を、
                #   GT 基準の量は `gt_` を頭に付ける。前回の混同は、両方が
                #   `picked_correct` という同じ名前だったことが直接の原因だった。
                yaw_diag = {
                    "candidate_scores": d["candidate_scores"],
                    "winner": w,
                    "margin": d["margin"],
                    "selfconsistency_correct_idx": int(np.argmin(errs)),
                    "selfconsistency_winner_rot_err_deg": round(errs[w], 3),
                    "selfconsistency_picked_correct": bool(int(np.argmin(errs)) == w),
                }
                if T_ref_or_none is not None:
                    R_exp_gt = metrics.decompose_sim3(
                        T_ref_or_none @ metrics.invert_sim3(P))[0]
                    egt = [metrics.rotation_error_deg(np.asarray(R), R_exp_gt)
                           for R in d["candidate_R"]]
                    yaw_diag.update({
                        "gt_correct_idx": int(np.argmin(egt)),
                        "gt_winner_rot_err_deg": round(egt[w], 3),
                        "gt_best_possible_rot_err_deg": round(float(min(egt)), 3),
                        "gt_picked_correct": bool(int(np.argmin(egt)) == w),
                    })
        if yaw_diag is not None:
            # ★ trials.csv には入れない。列が1つ増えるだけで既存の出力と
            #    バイト単位で一致しなくなり、T2 の回帰確認が通らなくなる。
            yaw_records.append(dict(yaw_diag, method=method_name, trial=i,
                                    selfconsistency_success=bool(ok),
                                    gt_success=ok_gt))
        rec = {
            "method": method_name,
            "trial": i,
            "pert_rot_deg": round(metrics.rotation_error_deg(Rp, np.eye(3)), 3),
            "pert_trans": round(float(np.linalg.norm(tp)), 4),
            "pert_scale": round(float(sp), 4),
            # ★ 旧列 `rot_deg` / `trans` / `scale_ratio` / `success` は
            #   どちらの基準か読み手に分からないので廃止した（R7 §3-1）。
            "selfconsistency_rot_deg": round(e["rot_deg"], 4),
            "selfconsistency_trans": round(e["trans"], 5),
            "selfconsistency_scale_ratio": round(e["scale_ratio"], 5),
            "selfconsistency_success": bool(ok),
            "selfconsistency_success_strict": ok_strict,
            "time_s": round(dt, 3),
        }
        if e_gt is not None:
            rec.update({
                "gt_rot_deg": round(e_gt["rot_deg"], 4),
                "gt_trans": round(e_gt["trans"], 5),
                "gt_scale_ratio": round(e_gt["scale_ratio"], 5),
                "gt_success": ok_gt,
            })
        trial_records.append(rec)
        # R7 §3-2: 行列そのものを残す。指標の定義を変えても**再実行せずに**測り直せる。
        # 今回のような取り違えが起きたとき、20,800 試行を回し直さずに済む保険である。
        matrix_records.append({
            "method": method_name, "trial": i,
            "T_est": np.asarray(Tt, dtype=np.float64).tolist(),
            "P": np.asarray(P, dtype=np.float64).tolist(),
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
        # ★ 旧 `success_rate` は廃止（R7 §3-1）。どちらの基準か名前で分かるようにする。
        "selfconsistency_success_rate": (round(float(np.mean(successes)), 3)
                                         if successes else float("nan")),
        "selfconsistency_med_rot_deg": round(med(rot_errs), 3),
        "selfconsistency_med_trans": round(med(trans_errs), 4),
        "selfconsistency_med_scale_ratio": round(med(scale_errs), 4),
        "med_time_s": round(med(times), 3),
    }

    # R7 §3-1: GT 基準の集計。T_gt が無ければ None（合成の凍結 T_gt のように
    # 手法自身の出力である場合は、値が出ても意味が無い点に注意）。
    if gt_successes:
        g_lo, g_hi = stats.wilson_ci(int(np.sum(gt_successes)), len(gt_successes))
        row["gt_success_rate"] = round(float(np.mean(gt_successes)), 3)
        row["gt_success_ci_lo"] = round(g_lo, 3)
        row["gt_success_ci_hi"] = round(g_hi, 3)
        row["gt_med_rot_deg"] = round(med(gt_rot), 3)
        row["gt_med_trans"] = round(med(gt_trans), 4)
        row["gt_med_scale_ratio"] = round(med(gt_scale_e), 4)
    else:
        for k in ("gt_success_rate", "gt_success_ci_lo", "gt_success_ci_hi",
                  "gt_med_rot_deg", "gt_med_trans", "gt_med_scale_ratio"):
            row[k] = None

    ci_lo, ci_hi = stats.wilson_ci(int(np.sum(successes)), len(successes))
    row["selfconsistency_success_ci_lo"] = round(ci_lo, 3)
    row["selfconsistency_success_ci_hi"] = round(ci_hi, 3)
    if successes_strict:
        s_lo, s_hi = stats.wilson_ci(int(np.sum(successes_strict)), len(successes_strict))
        row["selfconsistency_success_rate_strict"] = round(float(np.mean(successes_strict)), 3)
        row["selfconsistency_success_strict_ci_lo"] = round(s_lo, 3)
        row["selfconsistency_success_strict_ci_hi"] = round(s_hi, 3)
    else:
        row["selfconsistency_success_rate_strict"] = None
        row["selfconsistency_success_strict_ci_lo"] = None
        row["selfconsistency_success_strict_ci_hi"] = None
    for key, errs in (("rot_deg", rot_errs), ("trans", trans_errs), ("scale_ratio", scale_errs)):
        pct = stats.error_percentiles(errs)
        for stat_name in ("min", "q25", "q75", "max", "mean", "std"):
            row[f"selfconsistency_{key}_{stat_name}"] = round(pct[stat_name], 5)
    for key, errs in (("rot_deg", gt_rot), ("trans", gt_trans), ("scale_ratio", gt_scale_e)):
        pct = stats.error_percentiles(errs)
        for stat_name in ("min", "q25", "q75", "max", "mean", "std"):
            row[f"gt_{key}_{stat_name}"] = (round(pct[stat_name], 5)
                                            if errs else None)
    row["perturb_scale_prescaled"] = gt_scale     # 旧名 gt_scale_prescaled は紛らわしい
    return row, trial_records, yaw_records, matrix_records


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
                   config_path: str, trials: int, out_dir: str,
                   extra_meta: Optional[Dict] = None) -> None:
    """summary.json: reproducibility metadata + per-method failure breakdown."""
    succ = cfg["eval"]["success"]
    succ_strict = cfg["eval"].get("success_strict")
    per_method = {}
    for row in rows:
        name = row["method"]
        mine = [t for t in all_trials if t["method"] == name]
        sc = [{"rot_deg": t["selfconsistency_rot_deg"],
               "trans": t["selfconsistency_trans"],
               "scale_ratio": t["selfconsistency_scale_ratio"]} for t in mine]
        entry = {"aggregate": row,
                 # 名前で基準が分かるようにする（R7 §3-4）
                 "selfconsistency_failure_breakdown": stats.failure_breakdown(sc, succ)}
        gt = [{"rot_deg": t["gt_rot_deg"], "trans": t["gt_trans"],
               "scale_ratio": t["gt_scale_ratio"]}
              for t in mine if "gt_rot_deg" in t]
        if gt:
            entry["gt_failure_breakdown"] = stats.failure_breakdown(gt, succ)
        if succ_strict:
            entry["selfconsistency_failure_breakdown_strict"] = \
                stats.failure_breakdown(sc, succ_strict)
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
    summary.update(extra_meta or {})
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


def _write_figures(rows: List[Dict], out_dir: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [r["method"] for r in rows]
    panels = [("selfconsistency_success_rate", "Self-consistency rate (NOT accuracy)"),
              ("gt_success_rate", "GT-referenced success rate"),
              ("direct_chamfer", "Direct chamfer (m)"),
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
    ap.add_argument("--out-dir", default=None,
                    help="override cfg eval.out_dir (keeps prior runs intact)")
    ap.add_argument("--label-noise", type=float, default=None,
                    help="fraction of source points whose GT label is replaced "
                         "by a random different class (sensitivity, G4-3)")
    ap.add_argument("--voxel-size", type=float, default=None,
                    help="override cfg preprocess.voxel_size (density sweep, "
                         "G4-2); pass 0 to disable downsampling")
    args = ap.parse_args()
    cfg = load_config(args.config)
    trials = args.trials if args.trials is not None else int(cfg["eval"]["trials"])
    seed = int(cfg["eval"]["seed"])
    if args.voxel_size is not None:
        cfg["preprocess"]["voxel_size"] = float(args.voxel_size)

    src = io_utils.load_source_cloud(cfg)
    dst = io_utils.load_reference_cloud(cfg)
    if args.label_noise:
        # Sensitivity (G4-3): corrupt a fixed fraction of source labels once,
        # deterministically (seed offset by the noise level), then run the usual
        # perturbation protocol on the corrupted cloud.
        noise_seed = seed + int(round(args.label_noise * 1000)) + 7919
        n_rng = np.random.default_rng(noise_seed)
        n = len(src.labels)
        n_flip = int(round(args.label_noise * n))
        idx = n_rng.choice(n, size=n_flip, replace=False)
        n_classes = 6
        offsets = n_rng.integers(1, n_classes, size=n_flip)
        src.labels[idx] = (src.labels[idx] + offsets) % n_classes
        print(f"label noise: flipped {n_flip}/{n} source labels "
              f"(p={args.label_noise}, noise_seed={noise_seed})")
    T_gt = load_t_gt(cfg)
    if T_gt is None:
        print("WARNING: no frozen T_gt; recovery is measured relative to each "
              "method's own direct result. Run establish_gt.py for absolute error.")

    methods = args.methods or available_methods()
    print(f"benchmarking methods: {methods}  (trials={trials}, seed={seed})")
    rows = []
    all_trials: List[Dict] = []
    all_yaw: List[Dict] = []
    all_mat: List[Dict] = []
    for name in methods:
        print(f"  -> {name}")
        row, trial_records, yaw_records, matrix_records = _evaluate(
            name, src, dst, T_gt, cfg, trials, seed)
        rows.append(row)
        all_trials.extend(trial_records)
        all_yaw.extend(yaw_records)
        all_mat.extend(matrix_records)

    out_dir = args.out_dir or cfg["eval"]["out_dir"]
    _write_csv(rows, os.path.join(out_dir, "results.csv"))
    _write_csv(all_trials, os.path.join(out_dir, "trials.csv"))
    if all_yaw:
        # 別ファイルに出す。既存 config では空なので**ファイル自体が生まれない**
        with open(os.path.join(out_dir, "yaw_diag.json"), "w") as f:
            json.dump(all_yaw, f, indent=1)
    # R7 §3-2: 行列と、使った T_gt の素性を残す。指標の定義が変わっても
    # ここから測り直せる（今回はそれが無くて 20,800 試行を回し直す話になった）。
    tgt_path = cfg["eval"].get("t_gt_path")
    tgt_meta = {"path": tgt_path, "sha256": None, "provenance": None}
    if tgt_path and os.path.exists(tgt_path):
        import hashlib
        with open(tgt_path, "rb") as f:
            raw = f.read()
        tgt_meta["sha256"] = hashlib.sha256(raw).hexdigest()
        try:
            tgt_meta["provenance"] = json.loads(raw).get("provenance")
        except Exception:
            pass
    with open(os.path.join(out_dir, "trial_matrices.json"), "w") as f:
        json.dump({"T_gt": tgt_meta,
                   "note": "trial=-1 は無摂動の解 T0（自己一貫性の基準）",
                   "trials": all_mat}, f)
    _write_summary(rows, all_trials, cfg, args.config, trials, out_dir,
                   extra_meta={"label_noise": args.label_noise,
                               "voxel_size_override": args.voxel_size})
    _write_figures(rows, out_dir)

    # 採用の基準は **GT 基準の成功率**。無いときだけ自己一貫性で並べ、その旨を残す。
    # 自己一貫性で採用すると「同じ誤答に安定して戻る手法」が勝ちうる。
    has_gt = any(r.get("gt_success_rate") is not None for r in rows)
    key = "gt_success_rate" if has_gt else "selfconsistency_success_rate"
    best = sorted(rows, key=lambda r: (-(r[key] or 0.0), r["direct_chamfer"]))[0]
    with open(os.path.join(out_dir, "adopted.json"), "w") as f:
        json.dump({"adopted_method": best["method"], "ranked_by": key,
                   "t_gt": tgt_meta, "ranking": rows}, f, indent=2)

    print("\n=== results ===")
    for r in rows:
        print(f"{r['method']:18s} selfconsistency={r['selfconsistency_success_rate']}  "
              f"gt={r['gt_success_rate']}  chamfer={r['direct_chamfer']}  "
              f"med_time={r['med_time_s']}s")
    print(f"(ranked by {key})")
    print(f"\nADOPTED: {best['method']}  (results in {out_dir}/)")


if __name__ == "__main__":
    main()
