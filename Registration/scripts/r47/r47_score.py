"""R47 の採点（sni-slam の環境）。**回す前に commit する。判定の言葉は出さない（S4）。**

主な検定（§4-3、有意水準 0.05/4 = 0.0125）。STSL（主）= 両面の BIM・設定 A（R46 で全体の失敗が最も少なかった変種）、
5 回の多数決（3 回以上が大きな失敗なら、その試行は失敗）。4 シーン × 10 試行 = 40 の対で正確 McNemar（両側）。
**指示書が固定する水準を書いていないので、R46 で失敗が出た組（in411 × ref411）を起点にして 1 要因だけを動かす**：
- C-ref(r)：入力 = in411（411 の床から切る）に固定。参照 ref411 ↔ ref410_411。n10 = ref411 だけ失敗、n01 = ref410_411 だけ失敗
- C-src(r)：参照 = ref411 に固定。入力 in411 ↔ in410_411。n10 = in411 だけ失敗、n01 = in410_411 だけ失敗
  r ∈ {0.5, 3}。もう一方の水準に固定した組（入力 in410_411 での C-ref、参照 ref410_411 での C-src）は記述として出す
C-rot（B-2）：STSL（主）の失敗（向きごとに多数決）の割合を、dev < 5° と dev ≥ 10° で比べる。Fisher の正確検定（両側、0.05）。
  単位は向き（8 入力 × 36）。dev = |((ψ + 45) mod 90) − 45|（BIM の壁は BIM 座標の軸に平行なので、ψ がそのまま壁の向き）
大きな失敗：無効（候補なし・前処理の失敗）・e_R ≥ 45°・e_Ω ≥ 1.0 m（G1。G2〜G8 も）。分類は R46 §13 と同じ
記述：案A・案A 縮尺固定・重心、STSL の副（室内面・両面 × 設定 A・B）と best4、シーン別の件数、角の数（source・BIM）、
両方式（案A と STSL 主）が成功した試行の e_Ω・e_R の対の差、入力の同一性（点の数と sha256）
"""
import glob, hashlib, json, os, sys
from collections import Counter, defaultdict
from math import isfinite
import numpy as np

REPO = os.path.expanduser("~/rizu/SNI-SLAM")
sys.path.insert(0, os.path.join(REPO, "Registration")); sys.path.insert(0, os.path.join(REPO, "Registration", "scripts"))
sys.path.insert(0, os.path.join(REPO, "Registration", "scripts", "r46_stsl")); sys.path.insert(0, os.path.join(REPO, "Registration", "scripts", "r47"))
os.chdir(REPO)
from failure_decomposition import provenance                     # noqa: E402
from r41_rescore import per_trial, real_criteria                  # noqa: E402
from scipy.special import comb as _comb                          # noqa: E402
from scipy.stats import fisher_exact                             # noqa: E402
from r47_common import b1_conditions, B2_INPUTS, YAWS, FLOOR     # noqa: E402

ROOT = os.environ.get("R46_STSL_ROOT", os.path.expanduser("~/rizu/r46_stsl_data"))
ALPHA = 0.05 / 4
STSL_VARS = {"inner_A": ("", "A"), "inner_B": ("", "B"), "both_A": ("_both", "A"), "both_B": ("_both", "B")}
MAIN = "both_A"
SNI = ("planA", "planA_fixed", "centroid")


def jl(line):
    import re
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


def stsl_runs(setname, floor, st, Gs, om, omb, S, spec, P_mode):
    runs = {i: [] for i in range(len(spec["trials"]))}
    for k in range(1, 6):
        trp = json.load(open(os.path.join(ROOT, "processed", "%s_%s_k%d" % (setname, st, k), "30m", "T_rp.json"))) \
            if os.path.exists(os.path.join(ROOT, "processed", "%s_%s_k%d" % (setname, st, k), "30m", "T_rp.json")) else None
        pt = json.load(open(os.path.join(ROOT, "results", setname, "%s_k%d_prep_time.json" % (st, k))))
        for r in (jl(l) for l in open(os.path.join(ROOT, "results", setname, "%s_k%d_%s.jsonl" % (st, k, floor)))):
            i = r["index"]; t = spec["trials"][i]
            Trp = np.asarray(trp[str(i)]) if trp is not None else np.eye(4)
            Tin = S @ np.asarray(r["tf"]).reshape(4, 4) @ Trp
            if P_mode == "b1":
                T, P = Tin @ np.asarray(t["A"]) @ np.asarray(t["Rg"]), None
            else:
                T, P = Tin, np.asarray(t["P_src_to_input"])
            valid = len(r["candidates"]) > 0 and isfinite(r["optimal_score"]) and i not in pt.get("prep_failed", [])
            per = {str(j): per_trial(T, P, G, om, omb) for j, G in Gs.items()}
            if not valid:
                per = {j: dict(p, v=False) for j, p in per.items()}
            c = r["candidates"]
            c0 = ("prep_failed" if i in pt.get("prep_failed", []) else "no_candidate" if not c else
                  ("chosen_correct" if r["success_stsl"] else ("correct_existed_other_chosen" if any(x["correct"] for x in c) else "no_correct_candidate")))
            runs[i].append({"per": per, "class": cls_of(per["1"], c0), "n_corners": r["n_corners"], "n_cand": len(c)})
    return [runs[i] for i in range(len(spec["trials"]))]


def maj_fail(runs, j="1"):
    return [sum(big(x["per"][j]) for x in rs) >= 3 for rs in runs]


def main():
    om_all = np.load("Registration/output/diag/omega.npz"); omb_all = np.load("Registration/output/diag/r39_omega_bim.npz")
    res = {"provenance": provenance(), "alpha": ALPHA, "b1": {}, "input_identity": {}}
    # ---------------- B-1
    for c in b1_conditions():
        s, ref = c["scene"], c["ref"]
        Gs = real_criteria(s); om = om_all[s]; omb = omb_all[c["ref_target"]]
        zf = json.load(open(os.path.join(ROOT, "BIM", FLOOR[ref], "meta.json")))["z_floor"]; S = np.eye(4); S[2, 3] = zf
        setname = "real_" + c["input_set"]
        spec = json.load(open(os.path.join(ROOT, "inputs", setname, "spec.json")))
        m = {}
        f = os.path.join(ROOT, "results_r47", "sni", c["name"] + ".json") if c["cross"] else \
            os.path.join(ROOT, "results_sni", "%s_r%s.json" % (c["ref_target"], c["cut"]))
        rows = json.load(open(f))["rows"]
        for meth in SNI:
            m[meth] = [{str(j): per_trial(rows["%s|%d" % (meth, k)]["T_est"], rows["%s|%d" % (meth, k)]["P"], G, om, omb) for j, G in Gs.items()}
                       for k in range(10)]
        for v, (suf, st) in STSL_VARS.items():
            m[v] = stsl_runs(setname, FLOOR[ref] + suf, st, Gs, om, omb, S, spec, "b1")
        fails = {meth: [big(t["1"]) for t in m[meth]] for meth in SNI}
        fails.update({v: maj_fail(m[v]) for v in STSL_VARS})
        best4 = min(STSL_VARS, key=lambda v: (sum(fails[v]), list(STSL_VARS).index(v)))
        fails["STSL_best4"] = fails[best4]
        crit = {meth: {str(j): (sum(big(t[str(j)]) for t in m[meth]) if meth in SNI else sum(maj_fail(m[meth], str(j)))) for j in range(1, 9)}
                for meth in list(SNI) + list(STSL_VARS)}
        diffs = []
        for k in range(10):
            if not fails["planA"][k] and not fails[MAIN][k]:
                val = [x["per"]["1"] for x in m[MAIN][k] if x["per"]["1"]["v"]]
                diffs.append({"dO": m["planA"][k]["1"]["eO"] - float(np.median([v["eO"] for v in val])),
                              "dR": m["planA"][k]["1"]["eR"] - float(np.median([v["eR"] for v in val]))})
        res["b1"][c["name"]] = {**{kk: c[kk] for kk in ("scene", "in", "ref", "cut", "cross", "input_set")}, "fails": fails, "best4": best4,
                                "n_fail": {k: int(sum(v)) for k, v in fails.items()}, "by_criterion": crit,
                                "classes_main": dict(Counter(x["class"] for rs in m[MAIN] for x in rs)),
                                "source_corners_main": [x["n_corners"] for rs in m[MAIN] for x in rs], "diffs": diffs,
                                "reused_from_r46": not c["cross"]}
        if c["input_set"] not in res["input_identity"]:
            z = np.load(os.path.join(ROOT, "inputs_sni", c["input_set"] + ".npz"))
            res["input_identity"][c["input_set"]] = {"n_points": int(len(z["points"])),
                                                     "sha256": hashlib.sha256(np.ascontiguousarray(z["points"]).tobytes()).hexdigest()}
        print(c["name"], {k: int(sum(v)) for k, v in fails.items()}, flush=True)
    # 主な検定
    B = res["b1"]; tests = {}
    for cut in ("0p5", "3"):
        def pairs(a, b, meth=MAIN):
            fa = [x for s in ("m3_cor_a", "m3_cor_b", "m3_cor_c", "m3_cor_d") for x in B["%s__%s_r%s" % (s, a, cut)]["fails"][meth]]
            fb = [x for s in ("m3_cor_a", "m3_cor_b", "m3_cor_c", "m3_cor_d") for x in B["%s__%s_r%s" % (s, b, cut)]["fails"][meth]]
            n10 = sum(x and not y for x, y in zip(fa, fb)); n01 = sum(y and not x for x, y in zip(fa, fb))
            return {"n_pairs": len(fa), "n_fail_first": sum(fa), "n_fail_second": sum(fb), "n10": n10, "n01": n01, "p": mcnemar(n10, n01)}
        tests["C-ref_r%s" % cut] = {"primary": True, "first": "in411__ref411", "second": "in411__ref410_411", **pairs("in411__ref411", "in411__ref410_411")}
        tests["C-src_r%s" % cut] = {"primary": True, "first": "in411__ref411", "second": "in410_411__ref411", **pairs("in411__ref411", "in410_411__ref411")}
        tests["C-ref_r%s_alt" % cut] = {"primary": False, "first": "in410_411__ref411", "second": "in410_411__ref410_411", **pairs("in410_411__ref411", "in410_411__ref410_411")}
        tests["C-src_r%s_alt" % cut] = {"primary": False, "first": "in411__ref410_411", "second": "in410_411__ref410_411", **pairs("in411__ref410_411", "in410_411__ref410_411")}
        for meth in ("planA", "planA_fixed", "centroid", "inner_A", "inner_B", "both_B", "STSL_best4"):
            tests["C-ref_r%s__%s" % (cut, meth)] = {"primary": False, **pairs("in411__ref411", "in411__ref410_411", meth)}
            tests["C-src_r%s__%s" % (cut, meth)] = {"primary": False, **pairs("in411__ref411", "in410_411__ref411", meth)}
    for t in tests.values():
        if t.get("primary"):
            t["below_alpha"] = t["p"] < ALPHA
    res["tests"] = tests
    for k, t in tests.items():
        if t.get("primary") or k.endswith("_alt"):
            print(k, t, flush=True)
    # ---------------- B-2
    b2 = []
    for s, cut in B2_INPUTS:
        Gs = {1: real_criteria(s)[1]}; om = om_all[s]; omb = omb_all["%s__E3" % s]
        S = np.eye(4); S[2, 3] = json.load(open(os.path.join(ROOT, "BIM", "M3_410_411", "meta.json")))["z_floor"]
        name = "r47rot_%s_%s" % (s, cut)
        spec = json.load(open(os.path.join(ROOT, "inputs", name, "spec.json")))
        runs = stsl_runs(name, "M3_410_411_both", "A", Gs, om, omb, S, spec, "b2")
        rows = json.load(open(os.path.join(ROOT, "results_r47", "sni", "%s__in410_411__ref410_411_r%s_rot.json" % (s, cut))))["rows"]
        for i, y in enumerate(YAWS):
            pa = per_trial(rows["planA|%d" % i]["T_est"], rows["planA|%d" % i]["P"], Gs[1], om, omb)
            dev = abs(((y + 45) % 90) - 45)
            b2.append({"scene": s, "cut": cut, "yaw": y, "dev": dev, "stsl_fail": sum(big(x["per"]["1"]) for x in runs[i]) >= 3,
                       "stsl_classes": dict(Counter(x["class"] for x in runs[i])), "source_corners_median": float(np.median([x["n_corners"] for x in runs[i]])),
                       "n_cand_median": float(np.median([x["n_cand"] for x in runs[i]])), "planA_fail": big(pa), "planA_eO": pa["eO"], "planA_eR": pa["eR"]})
    lo = [r["stsl_fail"] for r in b2 if r["dev"] < 5]; hi = [r["stsl_fail"] for r in b2 if r["dev"] >= 10]
    odds, p = fisher_exact([[sum(lo), len(lo) - sum(lo)], [sum(hi), len(hi) - sum(hi)]])
    res["b2"] = {"rows": b2, "fisher": {"dev_lt5": [sum(lo), len(lo)], "dev_ge10": [sum(hi), len(hi)], "odds_ratio": odds, "p": p, "alpha": 0.05},
                 "planA": {"dev_lt5": [sum(r["planA_fail"] for r in b2 if r["dev"] < 5), len(lo)],
                           "dev_ge10": [sum(r["planA_fail"] for r in b2 if r["dev"] >= 10), len(hi)]}}
    print("C-rot", res["b2"]["fisher"], "planA", res["b2"]["planA"])
    res["bim_counts"] = [json.loads(l) for l in open(os.path.join(ROOT, "results", "close", "bim_counts.jsonl"))]
    os.makedirs("Registration/output/diag/r47", exist_ok=True)
    json.dump(res, open("Registration/output/diag/r47/r47_score.json", "w"), ensure_ascii=False, default=float)
    print("wrote Registration/output/diag/r47/r47_score.json")


if __name__ == "__main__":
    main()
