"""R36 §4 — 回転を解放する短い精緻化（`proposed.rotation_release`）を、実データ 8 条件で測る。

段階
----
0. **§4-4 の確認**：選択肢 off で R35 の 16 実行をやり直し、$\\hat T$ が
   `r35_runs.json` とビット一致すること（`--skip-off` で省略）
1. 選択肢 on（`max_iter` 10、対応距離・Tukey は config のまま）で案A の 8 条件を回す
2. 採点：開始時（= R35 の最終解）と終了時について
   回転・傾き・ヨー・縮尺・$d_\\Omega$・2項分解（基準1）、6 基準の値と3区分、主基準の合否
3. §4-2 の既知の失敗（**判定規則はここで固定。結果を見て変えない**）
   - 縮尺の潰れ：`is_degenerate_sim3`、または 終了/開始 の縮尺比が 0.5 未満か 2 超
   - 180°・90° の取り違え：開始→終了の回転の変化が 45° 超
   - 正のフィードバック：反復を通じて対応数が増え（最終 > 初回）、残差が下がり（最終 < 初回）、
     **かつ** $d_\\Omega$（基準1）が開始時より大きくなる
4. §7-2：オラクル回転（R36 §3 の $T$）を 6 基準で採点し、R35／§4／オラクルを横に並べる

**判定（§4-3）はスクリプトでは出さない。** 表を出すだけで、判定は報告で行う。

    conda activate sni-slam
    python Registration/scripts/r36_release.py
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from typing import Dict, List

import numpy as np
import yaml
from scipy.spatial import cKDTree

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "Registration"))
sys.path.insert(0, os.path.join(REPO, "Registration", "scripts"))
os.chdir(REPO)

from criterion_verdict import d_omega                        # noqa: E402
from failure_decomposition import provenance                 # noqa: E402
from r35_rescore import decompose, register                  # noqa: E402
from r36_rotation_diag import tilt_yaw                       # noqa: E402
from regbim import io_utils, metrics, preprocess             # noqa: E402
from regbim.labels import NAME_TO_ID                         # noqa: E402
from regbim.methods import get_method                        # noqa: E402

PRIMARY = 1
MAX_ITER = 10
SCALE_COLLAPSE = (0.5, 2.0)
FLIP_DEG = 45.0
KIND = {"m3_cor_c__E3": "a", "m3_cor_a__E2": "b", "m3_cor_a__E3": "b"}


def criteria(spread, scene):
    return [(f["id"], np.asarray(f["T"], dtype=np.float64))
            for f in spread[scene]["results"] if "T" in f]


def score_T(T, crits, omega, th) -> Dict:
    R, _, s = metrics.decompose_sim3(T)
    per = []
    for jid, G in crits:
        RG, _, sG = metrics.decompose_sim3(G)
        e = dict(id=jid, **tilt_yaw(R, RG))
        e["d_omega"] = d_omega(T, G, omega)
        e["scale_ratio"] = float(abs(s / sG - 1.0))
        e["ok"] = bool(e["rot_deg"] < th["rot_deg"] and e["d_omega"] < th["trans"]
                       and e["scale_ratio"] < th["scale_ratio"])
        per.append(e)
    n = sum(p["ok"] for p in per)
    G1 = dict(crits)[PRIMARY]
    degen = bool(metrics.is_degenerate_sim3(T))
    p1 = next(p for p in per if p["id"] == PRIMARY)
    return {"per_criterion": per, "n_ok": n,
            "verdict": ("all_pass" if n == len(per) else
                        "all_fail" if n == 0 else "criterion_dependent"),
            "primary": dict(p1, degenerate=degen, success=bool(p1["ok"] and not degen)),
            "decomposition_primary": decompose(T, G1, omega),
            "scale": s}


def iter_stats(trace, src_p, dst_p, cfg) -> List[Dict]:
    """各反復の姿勢で、ICP と同じ規則の対応数と残差を数え直す（ICP 自体は触らない）。"""
    max_corr = float(cfg["semantic_icp"]["max_corr_dist"])
    c = float(cfg["semantic_icp"]["tukey_c"])
    match_ids = [NAME_TO_ID[n] for n in cfg["classes"]["match_classes"]]
    common = sorted(set(src_p.present_classes()) & set(dst_p.present_classes())
                    & set(match_ids))
    trees = {k: cKDTree(dst_p.points[dst_p.labels == k]) for k in common
             if (dst_p.labels == k).any()}
    out = []
    for it, T in trace:
        moved = metrics.apply_sim3(np.asarray(T), src_p.points)
        ds = []
        for k in common:
            if k not in trees:
                continue
            d, _ = trees[k].query(moved[src_p.labels == k], k=1, workers=-1)
            ds.append(d[d < max_corr])
        d = np.concatenate(ds) if ds else np.zeros(0)
        w = (1.0 - np.clip(d / c, 0, 1) ** 2) ** 2
        out.append({"iter": int(it), "n_corr": int(len(d)),
                    "mean_dist": float(d.mean()) if len(d) else None,
                    "weighted_mean_dist": float((w * d).sum() / w.sum()) if w.sum() else None})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--r35-runs", default="Registration/output/diag/r35_runs.json")
    ap.add_argument("--r36-diag", default="Registration/output/diag/r36_rotation_diag.json")
    ap.add_argument("--spread", default="Registration/output/diag/criterion_spread.json")
    ap.add_argument("--omega-cache", default="Registration/output/diag/omega.npz")
    ap.add_argument("--out", default="Registration/output/diag/r36_release.json")
    ap.add_argument("--skip-off", action="store_true")
    args = ap.parse_args()

    r35 = json.load(open(args.r35_runs))["rows"]
    r36 = {r["target"]: r for r in json.load(open(args.r36_diag))["rows"]}
    spread = json.load(open(args.spread))["scenes"]
    z = np.load(args.omega_cache)
    res: Dict = {"provenance": provenance(), "max_iter": MAX_ITER,
                 "rules": {"scale_collapse_ratio": SCALE_COLLAPSE, "flip_deg": FLIP_DEG,
                           "feedback": "n_corr 最終>初回 かつ 残差 最終<初回 かつ d_Ω 悪化"},
                 "off_check": [], "rows": []}

    def dump():
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)

    # 0. off でビット一致
    if not args.skip_off:
        for r in r35:
            target = "%s__%s" % (r["scene"], r["cond"])
            cfg0 = yaml.safe_load(open("Registration/configs/realdata/%s.yaml" % target))
            T = np.asarray(register(cfg0, r["mode"])["T_est"], dtype=np.float64)
            eq = bool(np.array_equal(T, np.asarray(r["T_est"], dtype=np.float64)))
            res["off_check"].append({"target": target, "mode": r["mode"], "bit_equal": eq,
                                     "max_abs": float(np.abs(T - np.asarray(r["T_est"])).max())})
            print("off %-14s %-15s bit=%s" % (target, r["mode"], eq), flush=True)
            dump()

    # 1〜4. on
    for r in [x for x in r35 if x["mode"] == "plan_correlate"]:
        scene = r["scene"]
        target = "%s__%s" % (scene, r["cond"])
        cfg0 = yaml.safe_load(open("Registration/configs/realdata/%s.yaml" % target))
        th = cfg0["eval"]["success"]
        cfg = copy.deepcopy(cfg0)
        cfg["source"] = dict(cfg["source"], seed=0)
        cfg.setdefault("proposed", {})["translation_init"] = "plan_correlate"
        cfg["proposed"]["rotation_release"] = {"enabled": True, "max_iter": MAX_ITER}
        src = io_utils.load_source_cloud(cfg)
        dst = io_utils.load_reference_cloud(cfg)
        m = get_method("proposed")
        T_end = np.asarray(m.register(src, dst, cfg), dtype=np.float64)
        rd = m.last_release_diag
        T_start = np.asarray(rd["T_before"], dtype=np.float64)
        start_is_r35 = bool(np.array_equal(T_start, np.asarray(r["T_est"])))

        omega = z[scene]
        crits = criteria(spread, scene)
        a, b = score_T(T_start, crits, omega, th), score_T(T_end, crits, omega, th)
        R0, _, s0 = metrics.decompose_sim3(T_start)
        R1, _, s1 = metrics.decompose_sim3(T_end)
        src_p, dst_p = preprocess.prepare(src, cfg), preprocess.prepare(dst, cfg)
        its = iter_stats(rd["trace"], src_p, dst_p, cfg)

        dstart = a["decomposition_primary"]["d_omega_m"]
        dend = b["decomposition_primary"]["d_omega_m"]
        rot_change = metrics.rotation_error_deg(R1, R0)
        fb = bool(its[-1]["n_corr"] > its[0]["n_corr"]
                  and its[-1]["weighted_mean_dist"] < its[0]["weighted_mean_dist"]
                  and dend > dstart)
        failures = {
            "scale_collapse": bool(metrics.is_degenerate_sim3(T_end)
                                   or not (SCALE_COLLAPSE[0] <= s1 / s0 <= SCALE_COLLAPSE[1])),
            "scale_ratio_end_over_start": float(s1 / s0),
            "flip": bool(rot_change > FLIP_DEG),
            "rotation_change_deg": float(rot_change),
            "positive_feedback": fb}

        orc = {}
        for k, o in r36[target]["oracle_rotation"].items():
            orc[k] = score_T(np.asarray(o["T"], dtype=np.float64), crits, omega, th)

        row = {"target": target, "kind": KIND.get(target, "c"),
               "start_is_r35_final": start_is_r35, "n_iter": len(rd["trace"]) - 1,
               "start": a, "end": b, "oracle": orc, "failures": failures,
               "iterations": its, "T_end": T_end.tolist()}
        res["rows"].append(row)
        dump()
        p0, p1 = a["primary"], b["primary"]
        print("on %-14s(%s) 回転 %.3f→%.3f 傾き %.3f→%.3f ヨー %+.3f→%+.3f "
              "dΩ %.4f→%.4f 縮尺 %.4f→%.4f 変化 %.2f° 反復 %d 失敗 %s"
              % (target, row["kind"], p0["rot_deg"], p1["rot_deg"], p0["tilt_deg"],
                 p1["tilt_deg"], p0["yaw_deg"], p1["yaw_deg"], dstart, dend,
                 p0["scale_ratio"], p1["scale_ratio"], rot_change, row["n_iter"],
                 {k: v for k, v in failures.items() if isinstance(v, bool) and v}),
              flush=True)
    print("wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
