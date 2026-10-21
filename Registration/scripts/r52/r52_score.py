"""R52 の入力の記録と採点（sni-slam の環境）。**回す前に commit する。** 指示書 §3-2・§3-4・§2。

入力の記録（§3-2）：4 スキャン × f ∈ {0.2, 0.4, 0.6, 0.8, 1.0}（1.0 は今の `run1` のメッシュ）
- 使ったフレームの数（TSDF・投影の report の n_frames_used）、メッシュの頂点の数、標本化した点の数（R49 と同じ config の source、seed 0）
- 標本化した点を G1 で BIM の座標に置き、xy が 411・410 の床の外形の内側にある点の数（外形は R50 の `r50_tilt.Outline` と同じ作り方：
  `m3_ifc_<space>.npz` の is_inner かつ floor の点、0.1 m の格子、closing 1、穴埋め）
採点（§3-4、R51 段階0 の `r51_stage0.py` と同じ関数と定義）
- 対象：f = 0.2〜0.8 は `~/rizu/r52_data/results/f<f>/`、f = 1.0 は R49 の完全（`~/rizu/r49_data/results/real/full/`）
- 正解 G1（G2〜G8 は感度）、Ω は `omega.npz`。正しい＝有効かつ e_R < 45°・e_Ω < 1.0 m
- 段（f）× 参照 × スキャン、段 × 参照、段ごとに：大きな失敗（Wilson 95%）、正しい候補を含む割合（4 候補・16 候補、3 区分）、
  Δρ の分布・Δρ = 0 の数・AUC（試行とスキャン単位のブートストラップ 2,000 回、seed 0。段ごと）、
  失敗の違い（ヨーの差の 5 区分・並進の差）、16 候補の採点 score_short での正しい候補の順位の分布（f = 1.0 は記録が無い）、
  成功した試行の e_R・e_Ω（中央値・95%）
- T1：f = 0.2 と 1.0 の大きな失敗を、同じスキャン・参照・k で対にした正確 McNemar（80 対、両側、水準 0.05）
出力：`Registration/output/diag/r52/r52_inputs.json`・`r52_score.json`、試行ごと `~/rizu/r52_data/trials.json`
"""
import json, os, sys
from collections import Counter
import numpy as np

REPO = os.path.expanduser("~/rizu/SNI-SLAM")
for p in ("Registration", "Registration/scripts", "Registration/scripts/r49", "Registration/scripts/r50", "Registration/scripts/r51",
          "Registration/scripts/r52"):
    sys.path.insert(0, os.path.join(REPO, p))
os.chdir(REPO)
from regbim import io_utils, metrics                            # noqa: E402
from regbim.labels import NAME_TO_ID                            # noqa: E402
from failure_decomposition import provenance                    # noqa: E402
from r41_rescore import per_trial                               # noqa: E402
import r49_run as RR                                            # noqa: E402
from r49_score import truth, big, mcnemar                       # noqa: E402
from r50_tilt import Outline                                    # noqa: E402
from r51_stage0 import ok, rel, auc, wilson, boot_trials, boot_scenes, q, yaw_bin, NB  # noqa: E402
import r52_run as R52                                           # noqa: E402

FR = {"0p2": 0.2, "0p4": 0.4, "0p6": 0.6, "0p8": 0.8, "1p0": 1.0}
SCANS = ["m3_cor_a", "m3_cor_b", "m3_cor_c", "m3_cor_d"]


def mesh_dir(scan, f):
    return "output/RealData/_TSDF/%s/%s/mesh" % (scan, "run1" if f == "1p0" else "r52_f" + f)


def inputs():
    outl = {}
    for sp in ("411", "410"):
        z = np.load("Registration/output/ifc/m3_ifc_%s.npz" % sp)
        outl[sp] = Outline(z["points"][(z["labels"] == NAME_TO_ID["floor"]) & z["is_inner"], :2])
    res = {}
    for scan in SCANS:
        G1 = truth("real", scan + "__E2")[0]["1"]
        for f in FR:
            d = mesh_dir(scan, f)
            tr = json.load(open(os.path.join(d, "final_mesh_semantic_tsdf_report.json")))
            pr = json.load(open(os.path.join(d, "final_mesh_semantic_projected_report.json")))
            cfg = R52.config(scan + "__E2", f) if f != "1p0" else RR.config("real", scan + "__E2", "full")
            src = io_utils.load_source_cloud(cfg)
            Q = metrics.apply_sim3(G1, src.points)
            res["%s|%s" % (scan, f)] = {"tsdf_frames": tr["n_frames_used"], "proj_frames": pr["n_frames_used"],
                                        "frames_total": tr.get("n_frames_total"), "mesh_vertices": tr["n_vertices"],
                                        "n_sampled": int(len(src.points)),
                                        "class_counts": {k: int((src.labels == v).sum()) for k, v in NAME_TO_ID.items()},
                                        "inside_411": int(outl["411"].inside(Q[:, :2]).sum()), "inside_410": int(outl["410"].inside(Q[:, :2]).sum())}
            print(scan, f, res["%s|%s" % (scan, f)], flush=True)
    return res


def trials():
    rows = []
    for f in FR:
        for target in RR.REAL:
            Gs, om, _ = truth("real", target); c = om.mean(0)
            p = (os.path.join(os.path.expanduser("~/rizu/r49_data/results/real/full"), target + ".json") if f == "1p0"
                 else os.path.join(R52.OUT, "f" + f, target + ".json"))
            res = json.load(open(p))["rows"]
            for k in range(10):
                r = res[str(k)]; d = r["diag"]; P = np.asarray(r["P"], dtype=np.float64)
                sc = np.asarray(d["scores"], dtype=float); order = list(np.argsort(-sc, kind="stable")); Ts = d["after_icp"]
                corr = {j: [ok(per_trial(T, P, G, om, None)) for T in Ts] for j, G in Gs.items()}
                corr16 = [ok(per_trial(T, P, Gs["1"], om, None)) for T in d["short"]]
                pf = per_trial(r["T_est"], P, Gs["1"], om, None)
                ranks = [i + 1 for i, ci in enumerate(order) if corr["1"][ci]]
                row = {"f": f, "target": target, "target_scene": target.split("__")[0], "ref": target.split("__")[1], "trial": k,
                       "fail": bool(big(pf)), "fail_by_G": {j: bool(big(per_trial(r["T_est"], P, G, om, None))) for j, G in Gs.items()},
                       "eR": pf["eR"], "eO": pf["eO"], "valid": pf["v"],
                       "rho_sorted": [float(sc[i]) for i in order], "drho": float(sc[order[0]] - sc[order[1]]),
                       "contains": {j: bool(any(v)) for j, v in corr.items()}, "rank_correct": ranks[0] if ranks else None,
                       "short16_contains": bool(any(corr16))}
                if "score_short" in d:
                    s16 = np.asarray(d["score_short"], dtype=float); o16 = list(np.argsort(-s16, kind="stable"))
                    r16 = [i + 1 for i, ci in enumerate(o16) if corr16[ci]]
                    row["rank16_correct"] = r16[0] if r16 else None
                if ranks:
                    a, y, t, s = rel(Ts[order[0]], Ts[order[ranks[0] - 1]], P, c)
                    row["top_vs_correct"] = {"rot_deg": a, "yaw_deg": y, "trans_m": t, "scale_rel": s}
                rows.append(row)
    return rows


def summarize(rows):
    n = len(rows); nf = sum(r["fail"] for r in rows); kc = sum(r["contains"]["1"] for r in rows); fails = [r for r in rows if r["fail"]]
    succ = [r for r in rows if not r["fail"]]
    return {"n": n, "n_fail": nf, "fail_wilson95": wilson(nf, n),
            "contains4": {"k": kc, "frac": kc / n, "wilson95": wilson(kc, n)},
            "contains16": {"k": sum(r["short16_contains"] for r in rows), "frac": sum(r["short16_contains"] for r in rows) / n},
            "origin16": {"none": sum(not r["short16_contains"] and not r["contains"]["1"] for r in rows),
                         "in16_not4": sum(r["short16_contains"] and not r["contains"]["1"] for r in rows),
                         "in4": kc},
            "rank_correct": dict(Counter(str(r["rank_correct"]) for r in rows)),
            "rank16_correct": dict(Counter(str(r.get("rank16_correct", "no_record")) for r in rows)),
            "drho_fail": q([r["drho"] for r in fails]), "drho_success": q([r["drho"] for r in succ]),
            "drho_zero": {"fail": sum(r["drho"] == 0 for r in fails), "success": sum(r["drho"] == 0 for r in succ)},
            "fail_top_vs_correct": {"n_with_correct": sum("top_vs_correct" in r for r in fails),
                                    "yaw_bins": dict(Counter(yaw_bin(r["top_vs_correct"]["yaw_deg"]) for r in fails if "top_vs_correct" in r)),
                                    "trans_m": q([r["top_vs_correct"]["trans_m"] for r in fails if "top_vs_correct" in r])},
            "success_eR": q([r["eR"] for r in succ]), "success_eO": q([r["eO"] for r in succ]),
            "fail_by_G": {j: sum(r["fail_by_G"][j] for r in rows) for j in sorted(rows[0]["fail_by_G"])},
            "contains_by_G": {j: sum(r["contains"][j] for r in rows) for j in sorted(rows[0]["contains"])}}


def main():
    os.makedirs("Registration/output/diag/r52", exist_ok=True)
    json.dump(inputs(), open("Registration/output/diag/r52/r52_inputs.json", "w"), indent=1)
    rows = trials()
    json.dump(rows, open(os.path.expanduser("~/rizu/r52_data/trials.json"), "w"), default=float)
    res = {"provenance": provenance(), "by_f": {}, "by_f_ref": {}, "by_f_ref_scan": {}}
    for f in FR:
        rf = [r for r in rows if r["f"] == f]
        e = summarize(rf)
        stat_auc = lambda rs: auc([r["drho"] for r in rs if r["fail"]], [r["drho"] for r in rs if not r["fail"]])
        e["auc"] = {"value": stat_auc(rf), "trial_boot": boot_trials(rf, stat_auc, np.random.default_rng(0)),
                    "scan_boot": boot_scenes(rf, stat_auc, np.random.default_rng(0))}
        res["by_f"][f] = e
        for ref in ("E2", "E3"):
            res["by_f_ref"]["%s|%s" % (f, ref)] = summarize([r for r in rf if r["ref"] == ref])
            for scan in SCANS:
                res["by_f_ref_scan"]["%s|%s|%s" % (f, ref, scan)] = summarize([r for r in rf if r["ref"] == ref and r["target_scene"] == scan])
        print(f, "fail", e["n_fail"], "contains4", e["contains4"]["k"], "auc", e["auc"]["value"], flush=True)
    a = {(r["target"], r["trial"]): r["fail"] for r in rows if r["f"] == "0p2"}
    b = {(r["target"], r["trial"]): r["fail"] for r in rows if r["f"] == "1p0"}
    pairs = [(a[k], b[k]) for k in a]
    n10 = sum(x and not y for x, y in pairs); n01 = sum(y and not x for x, y in pairs)
    res["T1"] = {"n_pairs": len(pairs), "fail_0p2": sum(x for x, _ in pairs), "fail_1p0": sum(y for _, y in pairs),
                 "n10_0p2_only": n10, "n01_1p0_only": n01, "p": mcnemar(n10, n01), "alpha": 0.05}
    print("T1", res["T1"])
    json.dump(res, open("Registration/output/diag/r52/r52_score.json", "w"), ensure_ascii=False, indent=1, default=float)


if __name__ == "__main__":
    main()
