"""R32 §2-1 — 廊下 4 シーン × E2/E3 で、既定と案A を比較する。

**出す 6 量**（R32 §2-1）：

| # | 量 |
|---|---|
| 1 | 段階1の並進誤差（**0.39 m 未満に入った条件数を明記**） |
| 2 | 最終 $d_\Omega$ |
| 3 | 成功判定（事前登録の基準） |
| 4 | 最終の回転 |
| 5 | 段階別の全候補（R29 §3） |
| 6 | 時間の内訳と超過数（R29 §2-1 の 300 秒） |

**加えて R32 §3 の対応一致率を、同じ実行から出す。**

**既定は変えない。** `translation_init` はコマンドラインで上書きするだけである。

    conda activate sni-slam
    python Registration/scripts/plan_a_compare.py
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
from regbim.labels import NAME_TO_ID                        # noqa: E402
from regbim.methods import get_method                       # noqa: E402

SUCC = {"rot_deg": 5.0, "trans": 0.1, "scale_ratio": 0.05}   # 事前登録。動かさない
TARGET = 0.39          # R32 §1 の主基準（**但し書き3点つき**）
TIME_LIMIT = 300.0     # R29 §2-1
ROT_TOL = 5.0


def correspondence_agreement(src, dst, T_est, T_gt, thresh) -> Dict:
    """R32 §3 — **正解姿勢での対応と、実際の姿勢での対応の一致率。**

    **診断用である。手法にも選択器にも渡さない。**
    **正解姿勢での対応は評価側だけが持つ。**
    """
    from scipy.spatial import cKDTree
    common = set(np.unique(src.labels)) & set(np.unique(dst.labels))
    n_gt = n_est = n_both = n_same = 0
    for c in common:
        sm, dm = src.labels == c, dst.labels == c
        if sm.sum() == 0 or dm.sum() == 0:
            continue
        tree = cKDTree(dst.points[dm])
        d_gt, i_gt = tree.query(metrics.apply_sim3(T_gt, src.points[sm]), k=1,
                                workers=-1)
        d_es, i_es = tree.query(metrics.apply_sim3(T_est, src.points[sm]), k=1,
                                workers=-1)
        ok_gt, ok_es = d_gt < thresh, d_es < thresh
        both = ok_gt & ok_es
        n_gt += int(ok_gt.sum())
        n_est += int(ok_es.sum())
        n_both += int(both.sum())
        n_same += int((i_gt[both] == i_es[both]).sum())
    return {"n_corr_at_gt": n_gt, "n_corr_at_est": n_est,
            "n_corr_both": n_both, "n_same_target": n_same,
            "agreement": (n_same / n_both) if n_both else None}


def run(target: str, mode: str, omega, G, cfg0) -> Dict:
    cfg = copy.deepcopy(cfg0)
    cfg["source"] = dict(cfg["source"], seed=0)
    cfg.setdefault("proposed", {})["translation_init"] = mode
    cfg.setdefault("diagnostics", {})
    cfg["diagnostics"]["record_yaw"] = True
    cfg["diagnostics"]["record_stages"] = True

    t_load = time.time()
    src = io_utils.load_source_cloud(cfg)
    dst = io_utils.load_reference_cloud(cfg)
    t_load = time.time() - t_load

    RG = metrics.decompose_sim3(G)[0]
    m = get_method("proposed")
    t0 = time.time()
    T_final = m.register(src, dst, cfg)
    t_reg = time.time() - t0
    st = m.last_stage_diag
    plan = getattr(m, "last_plan_diag", None)

    # 段階1：正解の向きを持つ候補の種の誤差
    seeds = st["seeds"] if st and st.get("seeds") else []
    if not seeds and plan:
        seeds = [c["T_gen"] for c in plan.get("stage_candidates", [])]
    rots = [metrics.rotation_error_deg(
        metrics.decompose_sim3(np.asarray(t))[0], RG) for t in seeds] or [999.0]
    k = int(np.argmin(rots))
    stage1 = d_omega(np.asarray(seeds[k]), G, omega) if seeds else float("nan")

    e = metrics.sim3_errors(T_final, G)
    fin = d_omega(T_final, G, omega)
    ok = (not e["degenerate"] and e["rot_deg"] < SUCC["rot_deg"]
          and fin < SUCC["trans"] and e["scale_ratio"] < SUCC["scale_ratio"])
    agree = correspondence_agreement(src, dst, T_final, G,
                                     float(cfg["semantic_icp"]["max_corr_dist"]))
    return {"target": target, "mode": mode,
            "stage1_seed_d_omega_m": float(stage1),
            "stage1_under_target": bool(stage1 < TARGET),
            "correct_rot_candidate_deg": float(rots[k]),
            "final_d_omega_m": float(fin),
            "final_rot_deg": float(e["rot_deg"]),
            "final_scale_ratio": float(e["scale_ratio"]),
            "success": bool(ok),
            "time_load_s": round(t_load, 2), "time_register_s": round(t_reg, 2),
            "time_total_s": round(t_load + t_reg, 2),
            "over_time_limit": bool(t_load + t_reg > TIME_LIMIT),
            "correspondence_agreement": agree,
            "n_seeds": len(seeds),
            "plan_diag": ({k2: v for k2, v in plan.items()
                           if k2 != "stage_candidates"} if plan else None),
            "stage_candidates": (plan or {}).get("stage_candidates")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", nargs="*",
                    default=["m3_cor_a", "m3_cor_b", "m3_cor_c", "m3_cor_d"])
    ap.add_argument("--conds", nargs="*", default=["E2", "E3"])
    ap.add_argument("--modes", nargs="*", default=["centroid", "plan_correlate"])
    ap.add_argument("--gt-kit", default="output/GT_alignment_probe")
    ap.add_argument("--out", default="Registration/output/diag/plan_a_compare.json")
    args = ap.parse_args()

    rows: List[Dict] = []
    print("%-16s %-16s %10s %6s %10s %9s %6s %8s %8s"
          % ("条件", "方式", "段階1[m]", "<0.39", "最終[m]", "最終回転",
             "成功", "一致率", "時間[s]"))
    print("-" * 100)
    for scene in args.scenes:
        for cond in args.conds:
            target = "%s__%s" % (scene, cond)
            path = "Registration/configs/realdata/%s.yaml" % target
            if not os.path.exists(path):
                print("%-16s **config が無い**" % target)
                continue
            cfg0 = yaml.safe_load(open(path))
            c = dict(cfg0, source=dict(cfg0["source"], seed=0))
            omega = build_omega(c)
            G = np.asarray(json.load(open(os.path.join(
                args.gt_kit, "T_gt", "T_gt_%s.json" % scene)))["T_gt"],
                dtype=np.float64).reshape(4, 4)
            for mode in args.modes:
                try:
                    r = run(target, mode, omega, G, cfg0)
                except Exception as ex:
                    r = {"target": target, "mode": mode,
                         "error": "%s: %s" % (type(ex).__name__, ex)}
                    print("%-16s %-16s **失敗** %s" % (target, mode, r["error"][:50]))
                else:
                    a = r["correspondence_agreement"]["agreement"]
                    print("%-16s %-16s %10.3f %6s %10.3f %9.2f %6s %8s %8.1f"
                          % (target, mode, r["stage1_seed_d_omega_m"],
                             "○" if r["stage1_under_target"] else "×",
                             r["final_d_omega_m"], r["final_rot_deg"],
                             "○" if r["success"] else "×",
                             "—" if a is None else "%.3f" % a,
                             r["time_total_s"]))
                rows.append(r)
                os.makedirs(os.path.dirname(args.out), exist_ok=True)
                with open(args.out, "w") as f:
                    json.dump({"provenance": provenance(),
                               "success_thresholds": SUCC, "target_m": TARGET,
                               "time_limit_s": TIME_LIMIT, "rows": rows},
                              f, indent=2, ensure_ascii=False)

    print("\n## まとめ")
    for mode in args.modes:
        ok = [r for r in rows if r.get("mode") == mode and "error" not in r]
        if not ok:
            continue
        n1 = sum(1 for r in ok if r["stage1_under_target"])
        print("\n### %s（%d 条件）" % (mode, len(ok)))
        print("  段階1 < %.2f m       : **%d / %d 条件**" % (TARGET, n1, len(ok)))
        print("  段階1 の範囲          : %.3f 〜 %.3f m"
              % (min(r["stage1_seed_d_omega_m"] for r in ok),
                 max(r["stage1_seed_d_omega_m"] for r in ok)))
        print("  最終 d_Ω の範囲       : %.3f 〜 %.3f m"
              % (min(r["final_d_omega_m"] for r in ok),
                 max(r["final_d_omega_m"] for r in ok)))
        print("  成功                  : %d / %d"
              % (sum(1 for r in ok if r["success"]), len(ok)))
        ag = [r["correspondence_agreement"]["agreement"] for r in ok
              if r["correspondence_agreement"]["agreement"] is not None]
        if ag:
            print("  対応一致率            : 中央 %.3f / 範囲 %.3f 〜 %.3f"
                  % (float(np.median(ag)), min(ag), max(ag)))
        print("  時間                  : 中央 %.1f s / 最大 %.1f s / **超過 %d 件**"
              % (float(np.median([r["time_total_s"] for r in ok])),
                 max(r["time_total_s"] for r in ok),
                 sum(1 for r in ok if r["over_time_limit"])))

    print("\n**R32 §2-2 の判定境界**（結果を見てから動かさない）：")
    print("  8 条件すべて 1.0 m 未満 / 一部 / すべて 5 m 超")
    print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
