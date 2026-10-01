"""R39 §5 段階1 の採点。正しい形 C：d_Ω(T P, G, Ω)。回転 5°・0.1 m・縮尺 0.05、縮退は失敗（主基準と同じ）。

Ω：BIM 由来の source の構造点（`criterion_verdict.build_omega`、points_ply なので標本化なし、2 万点に決定的に間引く）。
事前登録（R39 §5-4、main が固定）：**案A（plan_correlate）の E2 相当**の成功数で
    95/100 以上 → 合う、50/100 未満 → 合わない、それ以外 → 中間
ほかの手法・E3 相当は記述。失敗した試行は 2 項分解（重心のずれ／重心まわり）と、回転・縮尺・並進のどれで落ちたかを出す。
"""
import json, os, sys
from collections import defaultdict
import numpy as np, yaml
os.chdir(os.path.expanduser("~/rizu/SNI-SLAM")); sys.path.insert(0, "Registration"); sys.path.insert(0, "Registration/scripts")
from criterion_verdict import build_omega, d_omega
from failure_decomposition import provenance
from r35_rescore import decompose
from r37_summary import wilson
from regbim import metrics

TH = {"rot_deg": 5.0, "trans": 0.1, "scale_ratio": 0.05}
RUNS = [("E2", "centroid"), ("E2", "plan_correlate"), ("E3", "centroid"), ("E3", "plan_correlate")]


def q(a):
    a = np.asarray([x for x in a if np.isfinite(x)], dtype=float)
    return None if not len(a) else {"median": float(np.median(a)), "p90": float(np.percentile(a, 90)),
                                    "max": float(a.max()), "min": float(a.min())}


def main():
    res = {"provenance": provenance(), "runs": {}}
    for ref, mode in RUNS:
        cfg = yaml.safe_load(open("Registration/configs/r39_bim/bim_%s_%s.yaml" % (ref, mode)))
        om = build_omega(cfg)
        base = "output/Registration/r39_bim/%s_%s" % (ref, mode)
        p = os.path.join(base, "trial_matrices.json")
        if not os.path.exists(p):
            print("%s_%s: まだ無い" % (ref, mode)); continue
        tm = json.load(open(p))
        G = np.asarray(json.load(open(cfg["eval"]["t_gt_path"]))["T_gt"], dtype=np.float64).reshape(4, 4)
        rows = defaultdict(list)
        for t in tm["trials"]:
            T = np.asarray(t["T_est"], dtype=np.float64)
            P = np.eye(4) if t["P"] is None else np.asarray(t["P"], dtype=np.float64)
            Tc = T @ P
            e = metrics.sim3_errors(Tc, G)
            if e.get("non_finite"):
                d = float("inf"); dec = None
            else:
                d = d_omega(Tc, G, om); dec = decompose(Tc, G, om)
            fr, fs, ft = e["rot_deg"] >= TH["rot_deg"], e["scale_ratio"] >= TH["scale_ratio"], d >= TH["trans"]
            ok = (not e["degenerate"]) and not (fr or fs or ft)
            rows[t["method"]].append({"trial": t["trial"], "rot_deg": e["rot_deg"], "scale_ratio": e["scale_ratio"],
                                      "d": d, "degenerate": bool(e["degenerate"]), "ok": bool(ok),
                                      "fail_rot": bool(fr), "fail_scale": bool(fs), "fail_trans": bool(ft),
                                      "decomp": dec})
        out = {}
        print("\n## %s 相当・%s（%s、commit %s）" % (ref, mode, base, tm.get("provenance", {}).get("commit", "?")[:7] if isinstance(tm.get("provenance"), dict) else "?"))
        for m, v in rows.items():
            v100 = [x for x in v if x["trial"] >= 0]
            k = sum(x["ok"] for x in v100); n = len(v100); lo, hi = wilson(k, n)
            fails = [x for x in v100 if not x["ok"]]
            out[m] = {"n": n, "k": k, "ci": [lo, hi],
                      "direct_T0": next((x for x in v if x["trial"] < 0), None),
                      "rot": q([x["rot_deg"] for x in v100]), "d": q([x["d"] for x in v100]),
                      "scale": q([x["scale_ratio"] for x in v100]),
                      "fail_counts": {"rot": sum(x["fail_rot"] for x in fails), "scale": sum(x["fail_scale"] for x in fails),
                                      "trans": sum(x["fail_trans"] for x in fails), "degenerate": sum(x["degenerate"] for x in fails),
                                      "rot_gt_45": sum(1 for x in fails if x["rot_deg"] > 45)},
                      "rows": v}
            print("%-22s 成功 %3d/%d [%.1f, %.1f]%% | 回転 中央 %.3f° 最大 %.2f° | dΩ 中央 %.4f 90%% %.4f 最大 %.3f | 縮尺 中央 %.4f | 失敗の内訳 %s"
                  % (m, k, n, 100 * lo, 100 * hi, out[m]["rot"]["median"], out[m]["rot"]["max"],
                     out[m]["d"]["median"], out[m]["d"]["p90"], out[m]["d"]["max"], out[m]["scale"]["median"], out[m]["fail_counts"]))
        res["runs"]["%s_%s" % (ref, mode)] = out
    r = res["runs"].get("E2_plan_correlate", {}).get("proposed")
    if r:
        k = r["k"]
        v = "合う" if k >= 95 else ("合わない" if k < 50 else "中間")
        res["verdict"] = {"k": k, "n": r["n"], "row": v}
        print("\n## 判定（案A・E2 相当）：%d/%d → %s" % (k, r["n"], v))
    json.dump(res, open("Registration/output/diag/r39_bim_score.json", "w"), ensure_ascii=False, default=float)


if __name__ == "__main__":
    main()
