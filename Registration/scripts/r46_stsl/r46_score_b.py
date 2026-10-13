"""R46 §10-2：B1（BIM 入力を両面から）と B2（入力の壁が軸に近い試行を除く）の採点（sni-slam の環境）。

**B1 は回す前に、B2 は段階1の結果を見た後の解析として、この規則を固定する。**
- STSL の変種：BIM 入力 {室内面 inner、両面 both} × 設定 {A, B}。大きな失敗の定義・多数決（5 回中 3 回）・無効の扱いは段階1と同じ
- STSL_best2：段階1と同じ（inner の A・B から条件ごとに良い方）。STSL_best4：4 変種から条件ごとに良い方
  （大きな失敗（多数決）が少ない方。同数なら inner_A, inner_B, both_A, both_B の順）。STSL に有利な扱い
- 案A・案A 縮尺固定・重心は段階1の値（`r46_score.json`）。**案A の参照は室内面のまま**（非対称）
- B2：試行ごとの「入力の壁の向きの、BIM の軸からのずれ」dev = |((ψ + 45) mod 90) − 45|、ψ は T_true（STSL の入力 → BIM）の
  ヨー。**dev ≥ 5° の試行だけ、dev ≥ 10° の試行だけ**に絞り、同じ表を作る。全方式を同じ試行に絞る。
  STSL の良い方の設定は、絞った試行の中で選び直す（STSL に有利）
- 出すもの：切り出しの段ごと・参照ごとの大きな失敗（件数と試行数）、対の比較（案A vs STSL_best2・best4、
  案A 縮尺固定 vs STSL_best4）の n10・n01 と条件ごとの正確 McNemar、STSL 自身の判定と候補なしの件数、失敗の分類
"""
import json, os, re, sys
from collections import defaultdict
from math import isfinite
import numpy as np

REPO = os.path.expanduser("~/rizu/SNI-SLAM")
sys.path.insert(0, os.path.join(REPO, "Registration")); sys.path.insert(0, os.path.join(REPO, "Registration", "scripts"))
os.chdir(REPO)
from failure_decomposition import provenance                     # noqa: E402
from r41_rescore import per_trial, real_criteria                  # noqa: E402
from scipy.special import comb as _comb                          # noqa: E402

ROOT = os.environ.get("R46_STSL_ROOT", os.path.expanduser("~/rizu/r46_stsl_data"))
CUTS = ["0p5", "3", "8", "inf"]
VARS = ["inner_A", "inner_B", "both_A", "both_B"]
FLOOR = {"E2": "M3_411", "E3": "M3_410_411"}
K = 5


def jl(line):
    return json.loads(re.sub(r"(?<![A-Za-z])(-?)inf(?![A-Za-z])", lambda m: m.group(1) + "Infinity", line).replace("nan", "NaN"))


def mcnemar(n10, n01):
    n = n10 + n01
    return 1.0 if n == 0 else min(1.0, 2.0 * sum(_comb(n, i, exact=True) for i in range(min(n10, n01) + 1)) / 2.0 ** n)


def big(p):
    return (not p["v"]) or p["eR"] >= 45.0 or p["eO"] >= 1.0


def load_both(name, ref, Gs, om, omb, spec, zf):
    S = np.eye(4); S[2, 3] = zf
    out = {}
    for st in "AB":
        runs = {i: [] for i in range(10)}
        for k in range(1, K + 1):
            seq = "real_%s_%s_k%d" % (name, st, k)
            trp = json.load(open(os.path.join(ROOT, "processed", seq, "30m", "T_rp.json")))
            pt = json.load(open(os.path.join(ROOT, "results", "real_" + name, "%s_k%d_prep_time.json" % (st, k))))
            for r in (jl(l) for l in open(os.path.join(ROOT, "results", "real_" + name, "%s_k%d_%s_both.jsonl" % (st, k, FLOOR[ref])))):
                i = r["index"]; t = spec["trials"][i]
                T = S @ np.asarray(r["tf"]).reshape(4, 4) @ np.asarray(trp[str(i)]) @ np.asarray(t["A"]) @ np.asarray(t["Rg"])
                valid = len(r["candidates"]) > 0 and isfinite(r["optimal_score"])
                per = {str(j): per_trial(T, None, G, om, omb) for j, G in Gs.items()}
                if not valid:
                    per = {j: dict(p, v=False) for j, p in per.items()}
                cand = r["candidates"]
                cls = ("prep_failed" if i in pt.get("prep_failed", []) else "no_candidate" if not cand else
                       ("chosen_correct" if r["success_stsl"] else
                        ("correct_existed_other_chosen" if any(c["correct"] for c in cand) else "no_correct_candidate")))
                runs[i].append({"per": per, "success_stsl": bool(r["success_stsl"]) and valid, "no_candidate": not cand, "class": cls})
        out["both_" + st] = [{"runs": runs[i]} for i in range(10)]
    return out


def fails(trials, stsl, j="1"):
    if stsl:
        return [sum(big(x["per"][j]) for x in t["runs"]) >= 3 for t in trials]
    return [big(t["per"][j]) for t in trials]


def main():
    sc = json.load(open("Registration/output/diag/r46/r46_score.json"))
    om_all = np.load("Registration/output/diag/omega.npz"); omb_all = np.load("Registration/output/diag/r39_omega_bim.npz")
    conds = {}
    for name, c in sc["conditions"].items():
        target, cut = name.rsplit("_r", 1); scene, ref = target.split("__")
        spec = json.load(open(os.path.join(ROOT, "inputs", "real_" + name, "spec.json")))
        zf = json.load(open(os.path.join(ROOT, "BIM", FLOOR[ref] + "_both", "meta.json")))["z_floor"]
        Gs = real_criteria(scene)
        meth = {m: c["trials"][m] for m in ("planA", "planA_fixed", "centroid")}
        meth["inner_A"] = c["trials"]["STSL_A"]; meth["inner_B"] = c["trials"]["STSL_B"]
        meth.update(load_both(name, ref, Gs, om_all[scene], omb_all[target], spec, zf))
        dev = []
        for t in spec["trials"]:
            Tt = np.asarray(t["T_true"]); y = np.degrees(np.arctan2(Tt[1, 0], Tt[0, 0]))
            dev.append(float(abs(((y + 45) % 90) - 45)))
        conds[name] = {"cut": cut, "ref": ref, "meth": meth, "dev": dev}
        print("loaded", name, flush=True)
    res = {"provenance": provenance(), "subsets": {}}
    for sub, thr in (("all", None), ("dev_ge5", 5.0), ("dev_ge10", 10.0)):
        tabs = {"by_cut": defaultdict(lambda: defaultdict(int)), "by_ref": defaultdict(lambda: defaultdict(int)),
                "pairs": defaultdict(lambda: defaultdict(int)), "conditions": {}}
        for name, c in conds.items():
            keep = [i for i in range(10) if thr is None or c["dev"][i] >= thr]
            if not keep:
                tabs["conditions"][name] = {"n_trials": 0}; continue
            f = {m: [x for i, x in enumerate(fails(c["meth"][m], m in VARS)) if i in keep] for m in c["meth"]}
            best2 = min(("inner_A", "inner_B"), key=lambda m: (sum(f[m]), VARS.index(m)))
            best4 = min(VARS, key=lambda m: (sum(f[m]), VARS.index(m)))
            f["STSL_best2"], f["STSL_best4"] = f[best2], f[best4]
            ent = {"n_trials": len(keep), "best2": best2, "best4": best4, "n_big_fail": {m: sum(v) for m, v in f.items()}, "pairs": {}}
            for a, b in (("planA", "STSL_best2"), ("planA", "STSL_best4"), ("planA_fixed", "STSL_best4")):
                n10 = sum(x and not y for x, y in zip(f[a], f[b])); n01 = sum(y and not x for x, y in zip(f[a], f[b]))
                ent["pairs"][a + "_vs_" + b] = {"n10": n10, "n01": n01, "p": mcnemar(n10, n01)}
                tabs["pairs"][a + "_vs_" + b]["n10"] += n10; tabs["pairs"][a + "_vs_" + b]["n01"] += n01
                tabs["pairs"][a + "_vs_" + b]["n_cond_p05"] += int(mcnemar(n10, n01) < 0.05)
            for m, v in f.items():
                tabs["by_cut"][c["cut"]][m] += sum(v); tabs["by_ref"][c["ref"]][m] += sum(v)
            tabs["by_cut"][c["cut"]]["N"] += len(keep); tabs["by_ref"][c["ref"]]["N"] += len(keep)
            tabs["conditions"][name] = ent
        res["subsets"][sub] = {k: (dict((kk, dict(vv)) for kk, vv in v.items()) if k != "conditions" else v) for k, v in tabs.items()}
        print(sub, {cut: dict(v) for cut, v in tabs["by_cut"].items()}, {p: dict(v) for p, v in tabs["pairs"].items()}, flush=True)
    # STSL 自身の判定・候補なし・分類（全試行）
    res["stsl_variants"] = {}
    for v in VARS:
        tr = [t for c in conds.values() for t in c["meth"][v]]
        res["stsl_variants"][v] = {
            "n_big_fail": sum(fails(tr, True)),
            "n_success_stsl_majority": sum(sum(x["success_stsl"] for x in t["runs"]) >= 3 for t in tr),
            "n_no_candidate_majority": sum(sum(x["no_candidate"] for x in t["runs"]) >= 3 for t in tr),
            "classes": {k: sum(x["class"] == k for t in tr for x in t["runs"]) for k in
                        ("chosen_correct", "prep_failed", "no_candidate", "correct_existed_other_chosen", "no_correct_candidate")}}
        print(v, res["stsl_variants"][v], flush=True)
    res["dev"] = {n: c["dev"] for n, c in conds.items()}
    json.dump(res, open("Registration/output/diag/r46/r46_score_b.json", "w"), ensure_ascii=False, default=float)
    print("wrote Registration/output/diag/r46/r46_score_b.json")


if __name__ == "__main__":
    main()
