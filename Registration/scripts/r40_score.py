"""R40 の採点（正しい形 C）と、事前登録の判定。

§3（E2'）判定（R40 §3-3）：対象は案A（plan_correlate）と既定（centroid）の `proposed`
    再現した：centroid < 50 かつ 案A ≥ 95 ／ 両方合う：両方 ≥ 95 ／ 両方失敗：両方 < 50 ／ それ以外は中間
§4 判定（R40 §4-4）：案A（`proposed`）と `no_semantic` の成功数の差（同じ入力・同じ摂動の 50 試行）
    判定に使う誤差は **O-wall と L**（R40 §4-3 の確認で、O-bg は no_semantic も使わない点なので記述のみ）
    支持：2 つ以上の誤差で、それぞれ 2 つ以上の水準において、差（案A − no_semantic）≥ 10 かつ Wilson 95% CI が重ならない
    支持されない：判定に使う誤差のすべての水準で、差（案A − no_semantic）≤ 3
    中間：それ以外
S（部分の回転）：解のヨー誤差 ψ（G に対する）が、0（411 に揃う）と、410 の回した角（410 に揃う）のどちらに近いか
"""
import json, os, sys
from collections import defaultdict
import numpy as np, yaml
os.chdir(os.path.expanduser("~/rizu/SNI-SLAM")); sys.path.insert(0, "Registration"); sys.path.insert(0, "Registration/scripts")
from criterion_verdict import build_omega, d_omega
from failure_decomposition import provenance
from r35_rescore import decompose
from r36_rotation_diag import tilt_yaw
from r37_summary import wilson
from regbim import metrics

TH = {"rot_deg": 5.0, "trans": 0.1, "scale_ratio": 0.05}
INPUTS = ["zero", "N_1cm", "N_2cm", "N_5cm", "L_5", "L_10", "L_20", "Obg_10", "Obg_25", "Obg_50",
          "Owall_10", "Owall_25", "Owall_50", "S_0p5", "S_1p0", "S_2p0"]


def q(a):
    a = np.asarray([x for x in a if np.isfinite(x)], dtype=float)
    return None if not len(a) else {"median": float(np.median(a)), "p90": float(np.percentile(a, 90)),
                                    "max": float(a.max()), "min": float(a.min())}


def score_run(name, cfg_name):
    cfg = yaml.safe_load(open("Registration/configs/r40_bim/%s.yaml" % cfg_name))
    # Ω は誤差なしの source の構造点で固定（入力によらず同じ点で測る。R40 §4 の比較のため）
    om = build_omega(yaml.safe_load(open("Registration/configs/r40_bim/err_zero.yaml")))
    p = "output/Registration/r40_bim/%s/trial_matrices.json" % name
    if not os.path.exists(p):
        return None
    tm = json.load(open(p))
    G = np.asarray(json.load(open(cfg["eval"]["t_gt_path"]))["T_gt"], dtype=np.float64).reshape(4, 4)
    RG = metrics.decompose_sim3(G)[0]
    out = defaultdict(list)
    for t in tm["trials"]:
        if t["trial"] < 0:
            continue
        T = np.asarray(t["T_est"], dtype=np.float64) @ np.asarray(t["P"], dtype=np.float64)
        e = metrics.sim3_errors(T, G)
        if e.get("non_finite"):
            out[t["method"]].append({"trial": t["trial"], "ok": False, "rot_deg": float("inf"), "d": float("inf"),
                                     "scale_ratio": float("inf"), "degenerate": True}); continue
        d = d_omega(T, G, om)
        ty = tilt_yaw(metrics.decompose_sim3(T)[0], RG)
        ok = (not e["degenerate"]) and e["rot_deg"] < TH["rot_deg"] and e["scale_ratio"] < TH["scale_ratio"] and d < TH["trans"]
        out[t["method"]].append({"trial": t["trial"], "ok": bool(ok), "rot_deg": e["rot_deg"], "tilt_deg": ty["tilt_deg"],
                                 "yaw_deg": ty["yaw_deg"], "d": d, "scale_ratio": e["scale_ratio"],
                                 "degenerate": bool(e["degenerate"]), "decomp": decompose(T, G, om),
                                 "fail_rot": e["rot_deg"] >= 5, "fail_scale": e["scale_ratio"] >= 0.05, "fail_trans": d >= 0.1})
    summ = {}
    for m, v in out.items():
        k, n = sum(x["ok"] for x in v), len(v)
        summ[m] = {"k": k, "n": n, "ci": wilson(k, n), "rot": q([x["rot_deg"] for x in v]),
                   "tilt": q([x.get("tilt_deg", np.inf) for x in v]), "yaw": q([abs(x.get("yaw_deg", np.inf)) for x in v]),
                   "yaw_signed_median": float(np.median([x.get("yaw_deg", np.nan) for x in v])),
                   "d": q([x["d"] for x in v]), "scale": q([x["scale_ratio"] for x in v]),
                   "fails": {"rot": sum(x.get("fail_rot", 0) for x in v if not x["ok"]),
                             "scale": sum(x.get("fail_scale", 0) for x in v if not x["ok"]),
                             "trans": sum(x.get("fail_trans", 0) for x in v if not x["ok"]),
                             "rot_gt_45": sum(1 for x in v if not x["ok"] and x["rot_deg"] > 45)},
                   "rows": v}
    return summ


def main():
    res = {"provenance": provenance(), "s3": {}, "s4": {}}
    print("## §3 E2'（参照 410）")
    for name in ("E2p_centroid", "E2p_plan_correlate"):
        s = score_run(name, name)
        res["s3"][name] = s
        if s:
            for m, x in s.items():
                print("%-20s %-22s %3d/%d [%.1f,%.1f]%% 回転中央 %.3f dΩ中央 %.4f 90%% %.4f 縮尺中央 %.4f 失敗 %s"
                      % (name, m, x["k"], x["n"], 100 * x["ci"][0], 100 * x["ci"][1], x["rot"]["median"],
                         x["d"]["median"], x["d"]["p90"], x["scale"]["median"], x["fails"]))
    try:
        kc = res["s3"]["E2p_centroid"]["proposed"]["k"]; ka = res["s3"]["E2p_plan_correlate"]["proposed"]["k"]
        v = ("再現した" if kc < 50 and ka >= 95 else "両方合う" if kc >= 95 and ka >= 95 else
             "両方失敗" if kc < 50 and ka < 50 else "中間")
        res["s3_verdict"] = {"centroid": kc, "plan_correlate": ka, "row": v}
        print("判定 §3：centroid %d / 案A %d → %s" % (kc, ka, v))
    except (KeyError, TypeError):
        pass
    print("\n## §4 誤差を 1 つずつ（参照 411、案A の config、50 試行）")
    for name in INPUTS:
        s = score_run("err_" + name, "err_" + name)
        res["s4"][name] = s
        if not s:
            print(name, "まだ無い"); continue
        print(name + "  " + " | ".join("%s %d/%d dΩ%.3f yaw%+.2f" % (m.replace("proposed", "P").replace("baseline_", "B_"),
                                                                   x["k"], x["n"], x["d"]["median"], x["yaw_signed_median"])
                                         for m, x in s.items()))
    # 判定
    judged = {"Owall": ["Owall_10", "Owall_25", "Owall_50"], "L": ["L_5", "L_10", "L_20"]}
    table = {}
    for err, levels in list(judged.items()) + [("Obg（記述のみ）", ["Obg_10", "Obg_25", "Obg_50"])]:
        for lv in levels:
            s = res["s4"].get(lv)
            if not s or "proposed" not in s or "proposed_no_semantic" not in s:
                continue
            a, b = s["proposed"], s["proposed_no_semantic"]
            diff = a["k"] - b["k"]
            sep = a["ci"][0] > b["ci"][1] or b["ci"][0] > a["ci"][1]
            table[lv] = {"planA": a["k"], "no_semantic": b["k"], "diff": diff, "ci_separate": bool(sep),
                         "support_level": bool(diff >= 10 and sep)}
            print("  %-10s 案A %2d  no_sem %2d  差 %+d  CI分離 %s" % (lv, a["k"], b["k"], diff, sep))
    res["c3_table"] = table
    n_err_support = sum(1 for err, lv in judged.items() if sum(table.get(x, {}).get("support_level", False) for x in lv) >= 2)
    all_small = all(table[x]["diff"] <= 3 for lv in judged.values() for x in lv if x in table)
    v = "支持" if n_err_support >= 2 else ("支持されない" if all_small else "中間")
    res["c3_verdict"] = {"n_errors_with_2plus_levels": n_err_support, "all_diff_le_3": all_small, "row": v}
    print("判定 §4（貢献3）：%s（2 水準以上で差≥10・CI分離の誤差の数 %d、判定の誤差の全水準で差≤3：%s）" % (v, n_err_support, all_small))
    json.dump(res, open("Registration/output/diag/r40_score.json", "w"), ensure_ascii=False, default=float)


if __name__ == "__main__":
    main()
