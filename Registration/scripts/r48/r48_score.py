"""R48 段階1 の採点（sni-slam の環境）。**回す前に commit する。** 判定の言葉は出さない（S4）。指示書 §5-6・§5-7 の通り。

試行の採点（R46・R47 と同じ）
- 案A・重心：source → 参照 = T_est · P。STSL：Shift(+z_floor) · tf · T_rp · A · Rg'（`r48_common.stsl_trial`）
- `r41_rescore.per_trial`（e_R・e_Ω（形 C の d_Ω、Ω はシーンごと：GT-A は `build_omega(gt_a/<scene>.yaml, seed 0)`、
  複数部屋は source の構造点 20,000 点、`r48_inputs.py`）・q）。G = M⁻¹（GT-A は R45 の T_gt）
- 大きな失敗：無効・e_R ≥ 45°・e_Ω ≥ 1.0 m。STSL は K = 5 の多数決（3 回以上）。STSL の無効 = 候補なし・optimal_score が有限でない・前処理の失敗
- 分類（R47 と同じ）：prep_failed／no_candidate／rot>=45:no_correct_candidate／rot>=45:correct_existed_other_chosen／
  orientation_ok_eO>=1m／success（STSL は K 回の各回で数える）
- GT-A の案A・重心は R45 の trial 0〜19
参照ごとの失敗の割合（範囲ごと、20 試行）。STSL の比較に入れる参照 = 被覆率 ≥ 0.80（`inputs_manifest.json`）
主な検定（水準 0.05/3）
- H1：STSL の失敗の割合（複数部屋は範囲 3 段の平均、GT-A は 1 段）と BIM 側の角の数の Spearman、片側（負：alternative="less"）
- H2：複数部屋の参照で、STSL の失敗の割合の差（切らない − r0.5）の Wilcoxon 符号順位、両側（scipy の既定：zero_method="wilcox"＝差 0 を除く）
- H3：「少ない」区分（角 ≤ 12）の参照 × 範囲ごとに、案A と STSL の失敗の割合を比べ、案A < STSL の組の数と案A > STSL の組の数の
  符号検定（二項、p = 0.5、両側。同じ割合の組は除く）
- H4（記述）：区分ごとの案A の失敗の割合（全試行をまとめた割合と Wilson 95% 区間）、参照ごとの案A の失敗の割合と角の数の Spearman（両側）
副の解析：dev < 5° の試行を除いた版で、同じ表と検定（主には使わない）。記述：試行単位の McNemar（案A 対 STSL）、e_Ω の分布、対の差
出力：`Registration/output/diag/r48/r48_score.json`
"""
import json, os, re, sys
from collections import Counter
from math import isfinite, sqrt
import numpy as np
import yaml
from scipy.stats import binom_test, spearmanr, wilcoxon                            # scipy 1.7

REPO = os.path.expanduser("~/rizu/SNI-SLAM")
sys.path.insert(0, os.path.join(REPO, "Registration")); sys.path.insert(0, os.path.join(REPO, "Registration", "scripts"))
sys.path.insert(0, os.path.join(REPO, "Registration", "scripts", "r48"))
os.chdir(REPO)
from failure_decomposition import provenance                    # noqa: E402
from r41_rescore import per_trial                               # noqa: E402
from scipy.special import comb as _comb                         # noqa: E402
import r48_common as C                                          # noqa: E402

ALPHA = 0.05 / 3
METHODS = ("planA", "centroid", "STSL")


def jl(line):
    return json.loads(re.sub(r"(?<![A-Za-z])(-?)inf(?![A-Za-z])", lambda m: m.group(1) + "Infinity", line).replace("nan", "NaN"))


def mcnemar(n10, n01):
    n = n10 + n01
    return 1.0 if n == 0 else min(1.0, 2.0 * sum(_comb(n, i, exact=True) for i in range(min(n10, n01) + 1)) / 2.0 ** n)


def big(p):
    return (not p["v"]) or p["eR"] >= 45.0 or p["eO"] >= 1.0


def cls_of(p, c):
    if not p["v"]:
        return c
    if p["eR"] >= 45:
        return "rot>=45:" + c
    if p["eO"] >= 1.0:
        return "orientation_ok_eO>=1m"
    return "success"


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return [None, None]
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [c - h, c + h]


def q(v):
    v = [x for x in v if x is not None and isfinite(x)]
    return None if not v else {"n": len(v), "p5": float(np.percentile(v, 5)), "p25": float(np.percentile(v, 25)),
                               "median": float(np.median(v)), "p75": float(np.percentile(v, 75)), "p95": float(np.percentile(v, 95))}


def stsl_runs(setname, floor, spec, G, om, zf, pc=False):
    S = np.eye(4); S[2, 3] = zf
    n = len(spec["trials"]); runs = {i: [] for i in range(n)}
    for k in range(1, 6):
        trp = json.load(open(os.path.join(C.D, "processed", "%s_A_k%d" % (setname, k), "30m", "T_rp.json")))
        pt = json.load(open(os.path.join(C.D, "results", setname, "A_k%d_prep_time.json" % k)))
        for r in (jl(l) for l in open(os.path.join(C.D, "results", setname, "A_k%d_%s.jsonl" % (k, floor)))):
            i = r["index"]; t = spec["trials"][i]
            Tin = S @ np.asarray(r["tf"]).reshape(4, 4) @ np.asarray(trp[str(i)])
            if pc:
                p = per_trial(Tin, None, np.asarray(t["T_true"]), om[i], None)
            else:
                p = per_trial(Tin @ np.asarray(t["A"]) @ np.asarray(t["Rg"]), None, G, om, None)
            pf = i in pt.get("prep_failed", [])
            if not (len(r["candidates"]) > 0 and isfinite(r["optimal_score"]) and not pf):
                p = dict(p, v=False)
            c = r["candidates"]
            c0 = ("prep_failed" if pf else "no_candidate" if not c else
                  ("chosen_correct" if r["success_stsl"] else ("correct_existed_other_chosen" if any(x["correct"] for x in c) else "no_correct_candidate")))
            runs[i].append({"per": p, "class": cls_of(p, c0), "n_corners": r["n_corners"], "n_cand": len(c)})
    return [runs[i] for i in range(n)]


def summarize_cell(cell, keep):
    """cell：方式 → 試行ごとの失敗（bool の列）。keep：使う試行の index"""
    return {m: (float(np.mean([cell[m][i] for i in keep])) if keep else None) for m in METHODS}


def tests(refs, man, counts, sub):
    """sub：試行を選ぶ関数 (ref, cut) → index の列"""
    out = {}
    inc = [r for r in refs if man["refs"][r]["stsl_included"]]
    rate = {(r, c): summarize_cell(refs[r]["cells"][c]["fails"], sub(r, c)) for r in refs for c in refs[r]["cells"]}
    def ref_rate(r, m):
        v = [rate[(r, c)][m] for c in refs[r]["cells"] if rate[(r, c)][m] is not None]
        return float(np.mean(v)) if v else None
    x = [counts[r] for r in inc if ref_rate(r, "STSL") is not None]; y = [ref_rate(r, "STSL") for r in inc if ref_rate(r, "STSL") is not None]
    rho, p = spearmanr(x, y, alternative="less")
    out["H1"] = {"n_refs": len(x), "spearman_rho": float(rho), "p_one_sided_less": float(p), "alpha": ALPHA}
    multi = [r for r in inc if not r.startswith("GTA_")]
    d = [rate[(r, "inf")]["STSL"] - rate[(r, "0p5")]["STSL"] for r in multi
         if rate[(r, "inf")]["STSL"] is not None and rate[(r, "0p5")]["STSL"] is not None]
    nz = [v for v in d if v != 0]
    if nz:
        w = wilcoxon(d, alternative="two-sided")
        out["H2"] = {"n_refs": len(d), "n_nonzero": len(nz), "n_pos_inf_more": sum(v > 0 for v in d), "n_neg_r05_more": sum(v < 0 for v in d),
                     "median_diff": float(np.median(d)), "statistic": float(w.statistic), "p_two_sided": float(w.pvalue), "alpha": ALPHA}
    else:
        out["H2"] = {"n_refs": len(d), "n_nonzero": 0, "p_two_sided": 1.0, "alpha": ALPHA}
    few = [(r, c) for r in inc if C.category(counts[r]) == "few" for c in refs[r]["cells"]
           if rate[(r, c)]["planA"] is not None and rate[(r, c)]["STSL"] is not None]
    lt = sum(rate[k]["planA"] < rate[k]["STSL"] for k in few); gt = sum(rate[k]["planA"] > rate[k]["STSL"] for k in few)
    out["H3"] = {"n_cells": len(few), "n_planA_less": lt, "n_planA_more": gt, "n_tie": len(few) - lt - gt,
                 "p_two_sided": float(binom_test(lt, lt + gt, 0.5)) if lt + gt else 1.0, "alpha": ALPHA}
    h4 = {}
    for cat in ("few", "mid", "many"):
        rs = [r for r in refs if C.category(counts[r]) == cat]
        k = n = 0
        for r in rs:
            for c in refs[r]["cells"]:
                idx = sub(r, c); k += sum(refs[r]["cells"][c]["fails"]["planA"][i] for i in idx); n += len(idx)
        h4[cat] = {"n_refs": len(rs), "n_trials": n, "n_fail": int(k), "rate": (k / n if n else None), "wilson95": wilson(k, n)}
    xa = [counts[r] for r in refs if ref_rate(r, "planA") is not None]; ya = [ref_rate(r, "planA") for r in refs if ref_rate(r, "planA") is not None]
    rho_a, p_a = spearmanr(xa, ya)
    h4["spearman_planA_rate_vs_corners"] = {"rho": (None if np.isnan(rho_a) else float(rho_a)), "p_two_sided": (None if np.isnan(p_a) else float(p_a))}
    out["H4"] = h4
    out["rates"] = {"%s|%s" % k: v for k, v in rate.items()}
    out["ref_rate"] = {r: {m: ref_rate(r, m) for m in METHODS} for r in refs}
    return out


def main():
    man = json.load(open(os.path.join(C.D, "inputs_manifest.json")))
    counts = {}
    for f in ("bim_counts.jsonl", "bim_counts_gta.jsonl"):
        for l in open(os.path.join(C.D, f)):
            j = json.loads(l); counts[j["floor"]] = j["n_corners"]
    om = dict(np.load(os.path.join(C.D, "omega.npz")))
    gta = np.load(os.path.join(C.D, "gta.npz")); Gm = np.asarray(man["G"])
    refs = {}; pcs = {}
    for ref in C.all_refs():
        scene = C.scene_of(ref); G = gta["G_" + scene] if ref.startswith("GTA_") else Gm; o = om[scene]
        zf = json.load(open(os.path.join(C.D, "BIM", ref, "meta.json")))["z_floor"]
        ent = {"scene": scene, "n_corners_bim": counts[ref], "category": C.category(counts[ref]),
               "coverage": man["refs"][ref]["coverage"], "stsl_included": man["refs"][ref]["stsl_included"], "cells": {}}
        for cut in C.cuts_of(ref):
            setname = "%s__%s" % (ref, cut)
            spec = json.load(open(os.path.join(C.D, "inputs", setname, "spec.json")))
            per = {}
            if ref.startswith("GTA_"):
                for m, d in (("planA", "plan_a"), ("centroid", "centroid")):
                    tm = json.load(open("output/Registration/r45/%s/%s/trial_matrices.json" % (d, scene)))
                    rows = {t["trial"]: t for t in tm["trials"] if t["trial"] >= 0}
                    per[m] = [per_trial(rows[k]["T_est"], rows[k]["P"], G, o, None) for k in range(C.N_TRIALS)]
            else:
                rows = json.load(open(os.path.join(C.D, "results_sni", setname + ".json")))["rows"]
                for m in ("planA", "centroid"):
                    per[m] = [per_trial(rows["%s|%d" % (m, k)]["T_est"], rows["%s|%d" % (m, k)]["P"], G, o, None) for k in range(C.N_TRIALS)]
            runs = stsl_runs(setname, ref, spec, G, o, zf)
            fails = {m: [big(p) for p in per[m]] for m in ("planA", "centroid")}
            fails["STSL"] = [sum(big(x["per"]) for x in rs) >= 3 for rs in runs]
            stsl_rep = []                          # 多数決の側の代表（失敗なら失敗した回の、成功なら成功した回の e_Ω の中央値に最も近い回）
            for rs, f in zip(runs, fails["STSL"]):
                side = [x for x in rs if big(x["per"]) == f]
                eos = [x["per"]["eO"] for x in side]
                stsl_rep.append(side[int(np.argsort(eos)[len(eos) // 2])]["per"])
            diffs = [{"dO": per["planA"][i]["eO"] - stsl_rep[i]["eO"], "dR": per["planA"][i]["eR"] - stsl_rep[i]["eR"]}
                     for i in range(C.N_TRIALS) if not fails["planA"][i] and not fails["STSL"][i]]
            ent["cells"][cut] = {
                "n_input_points": man["refs"][ref]["inputs"][cut]["n_points"],
                "dev": [t["dev"] for t in spec["trials"]], "fails": fails,
                "per": {m: [{k2: per[m][i][k2] for k2 in ("v", "eR", "eO", "q")} for i in range(C.N_TRIALS)] for m in ("planA", "centroid")},
                "stsl_rep": [{k2: p[k2] for k2 in ("v", "eR", "eO", "q")} for p in stsl_rep],
                "stsl_classes": dict(Counter(x["class"] for rs in runs for x in rs)),
                "stsl_source_corners": [int(np.median([x["n_corners"] for x in rs])) for rs in runs],
                "stsl_n_cand": [int(np.median([x["n_cand"] for x in rs])) for rs in runs],
                "diffs": diffs}
        refs[ref] = ent
        # 陽性対照（記述）
        sp = json.load(open(os.path.join(C.D, "inputs", "pc_" + ref, "spec.json")))
        rz = np.load(os.path.join(C.D, "refs", ref + ".npz")); RP = rz["points"].astype(np.float64)
        st = RP[np.isin(rz["labels"], [C.WALL, C.FLOOR, C.CEIL])]; st = st[np.linspace(0, len(st) - 1, 20000).astype(int)]
        oms = [C.apply(np.asarray(t["A"]), st) for t in sp["trials"]]
        pr = stsl_runs("pc_" + ref, ref, sp, None, oms, zf, pc=True)
        pf = [sum(big(x["per"]) for x in rs) >= 3 for rs in pr]
        pcs[ref] = {"n": len(pf), "n_fail": int(sum(pf)), "classes": dict(Counter(x["class"] for rs in pr for x in rs)),
                    "dev": [t["dev"] for t in sp["trials"]]}
        print(ref, ent["category"], counts[ref], {c: {m: sum(v["fails"][m]) for m in METHODS} for c, v in ent["cells"].items()},
              "pc fail %d/10" % pcs[ref]["n_fail"], flush=True)
    res = {"provenance": provenance(), "alpha": ALPHA, "refs": refs, "positive_control": pcs}
    res["main"] = tests(refs, man, counts, lambda r, c: list(range(C.N_TRIALS)))
    res["sub_dev_ge5"] = tests(refs, man, counts, lambda r, c: [i for i, d in enumerate(refs[r]["cells"][c]["dev"]) if d >= 5.0])
    # 記述：試行単位の McNemar（案A 対 STSL）、区分ごと・全体
    mc = {}
    for cat in ("few", "mid", "many", "all"):
        n10 = n01 = 0
        for r, e in refs.items():
            if cat != "all" and e["category"] != cat:
                continue
            for c in e["cells"].values():
                for a, b in zip(c["fails"]["planA"], c["fails"]["STSL"]):
                    n10 += a and not b; n01 += b and not a
        mc[cat] = {"n10_planA_only": n10, "n01_stsl_only": n01, "p": mcnemar(n10, n01)}
    res["mcnemar_trial"] = mc
    # 記述：e_Ω の分布（方式ごと・区分ごと。STSL は代表の回）、1.0〜1.2 m の数
    eo = {}
    for cat in ("few", "mid", "many", "all"):
        for m in METHODS:
            v = [(p["eO"] if p["v"] else None) for r, e in refs.items() if cat == "all" or e["category"] == cat
                 for c in e["cells"].values() for p in (c["stsl_rep"] if m == "STSL" else c["per"][m])]
            eo["%s|%s" % (cat, m)] = {"quantiles": q(v), "n_invalid": sum(x is None for x in v),
                                      "n_1_to_1p2": sum(x is not None and 1.0 <= x < 1.2 for x in v),
                                      "n_0p8_to_1": sum(x is not None and 0.8 <= x < 1.0 for x in v)}
    res["eO_dist"] = eo
    dd = [d for e in refs.values() for c in e["cells"].values() for d in c["diffs"]]
    res["paired_diffs"] = {"n": len(dd), "dO": q([d["dO"] for d in dd]), "dR": q([d["dR"] for d in dd]),
                           "n_planA_smaller_dO": sum(d["dO"] < 0 for d in dd), "n_planA_smaller_dR": sum(d["dR"] < 0 for d in dd)}
    os.makedirs("Registration/output/diag/r48", exist_ok=True)
    json.dump(res, open("Registration/output/diag/r48/r48_score.json", "w"), ensure_ascii=False, default=float)
    for k in ("H1", "H2", "H3"):
        print(k, res["main"][k]); print("  sub", res["sub_dev_ge5"][k])
    print("H4", res["main"]["H4"]); print("mcnemar", mc); print("diffs", res["paired_diffs"])


if __name__ == "__main__":
    main()
