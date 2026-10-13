"""R46 §13-2：閉じるための集計（保存済みの結果だけ。新しく回さない。sni-slam の環境）。

1. source 側の角の数：E2・E3 を別々に、切り出しの段ごと、設定 A・B ごと（`results/real_*/<設定>_k*_<参照>.jsonl` の n_corners。
   source 側の前処理は BIM の変種によらないので、室内面の版の結果を使う）。入力の点の数も並べる
2. BIM 側の角の数・三角形の記述子の数：`r46_bim_count`（`results/close/bim_counts.jsonl`）
3. best6（条件ごとに 6 変種から、大きな失敗（多数決）が最も少ないもの。同数なら inner_A, inner_B, both_A, both_B, axis_A, axis_B の順）
   の試行ごとの表：選ばれた変種、成否（多数決）、失敗の分類（5 回のうち最も多い分類。分類は候補なし／前処理の失敗／
   別を選んだ／正解に近い候補なし／向きは正しいが e_Ω ≥ 1 m（e_R < 45°・有効）／成功）
4. best6 の大きな失敗を G1〜G8 で
5. 両方式が成功した試行（案A は 1 回、STSL は多数決）での、e_Ω・e_R の対の差（案A − STSL）の分布。
   STSL の値は 5 回のうち有効な回の中央値
"""
import json, os, sys
from collections import Counter, defaultdict
import numpy as np

REPO = os.path.expanduser("~/rizu/SNI-SLAM")
sys.path.insert(0, os.path.join(REPO, "Registration")); sys.path.insert(0, os.path.join(REPO, "Registration", "scripts"))
sys.path.insert(0, os.path.join(REPO, "Registration", "scripts", "r46_stsl"))
os.chdir(REPO)
import r46_score_b as sb                                          # noqa: E402
from r41_rescore import real_criteria                            # noqa: E402
from failure_decomposition import provenance                     # noqa: E402

ROOT = sb.ROOT
VARS = sb.VARS
FLOOR = sb.FLOOR


def cls_of(x, j="1"):
    p = x["per"][j]
    if not p["v"]:
        return x["class"]                     # no_candidate / prep_failed
    if p["eR"] >= 45:
        return "rot>=45:" + x["class"]        # correct_existed_other_chosen / no_correct_candidate / chosen_correct(稀)
    if p["eO"] >= 1.0:
        return "orientation_ok_eO>=1m"
    return "success"


def main():
    sc = json.load(open("Registration/output/diag/r46/r46_score.json"))
    om_all = np.load("Registration/output/diag/omega.npz"); omb_all = np.load("Registration/output/diag/r39_omega_bim.npz")
    res = {"provenance": provenance(), "source_corners": {}, "conditions": {}}
    # 1. source 側の角
    for name in sc["conditions"]:
        target, cut = name.rsplit("_r", 1); ref = target.split("__")[1]
        z = np.load(os.path.join(ROOT, "inputs_sni", name + ".npz"))
        for st in "AB":
            cs = []
            for k in range(1, 6):
                for l in open(os.path.join(ROOT, "results", "real_" + name, "%s_k%d_%s.jsonl" % (st, k, FLOOR[ref]))):
                    cs.append(sb.jl(l)["n_corners"])
            res["source_corners"].setdefault("%s|%s|%s" % (ref, cut, st), []).extend(cs)
        res["source_corners"].setdefault("npts|%s|%s" % (ref, cut), []).append(int(len(z["points"])))
    # 3〜5
    g = defaultdict(lambda: defaultdict(int)); diffs = {"eO": [], "eR": []}; table = []
    for name, c in sc["conditions"].items():
        target, cut = name.rsplit("_r", 1); scene, ref = target.split("__")
        spec = json.load(open(os.path.join(ROOT, "inputs", "real_" + name, "spec.json")))
        zf = json.load(open(os.path.join(ROOT, "BIM", FLOOR[ref] + "_both", "meta.json")))["z_floor"]
        Gs = real_criteria(scene)
        meth = {"inner_A": c["trials"]["STSL_A"], "inner_B": c["trials"]["STSL_B"]}
        for v in ("both", "axis"):
            meth.update(sb.load_variant(name, ref, Gs, om_all[scene], omb_all[target], spec, zf, v))
        f = {m: sb.fails(meth[m], True) for m in VARS}
        best6 = min(VARS, key=lambda m: (sum(f[m]), VARS.index(m)))
        planA = c["trials"]["planA"]
        for j in map(str, range(1, 9)):
            g[ref][j] += sum(sb.fails(meth[best6], True, j)); g["all"][j] += sum(sb.fails(meth[best6], True, j))
        for i in range(10):
            runs = meth[best6][i]["runs"]
            fail = f[best6][i]
            cl = Counter(cls_of(x) for x in runs).most_common(1)[0][0]
            pa = planA[i]["per"]["1"]; pa_fail = sb.big(pa)
            table.append({"condition": name, "trial": i, "best6": best6, "stsl_fail": bool(fail), "stsl_class": cl,
                          "stsl_classes_5": dict(Counter(cls_of(x) for x in runs)), "planA_fail": bool(pa_fail)})
            if not fail and not pa_fail:
                val = [x["per"]["1"] for x in runs if x["per"]["1"]["v"]]
                diffs["eO"].append(pa["eO"] - float(np.median([v["eO"] for v in val])))
                diffs["eR"].append(pa["eR"] - float(np.median([v["eR"] for v in val])))
        res["conditions"][name] = {"best6": best6, "n_big_fail_best6": int(sum(f[best6])), "planA_fail": int(sum(sb.big(t["per"]["1"]) for t in planA))}
    res["best6_trials"] = table
    res["best6_big_fail_by_criterion"] = {k: dict(v) for k, v in g.items()}
    res["paired_diffs_both_success"] = {q: {"n": len(v), **({p: float(np.percentile(v, p)) for p in (5, 25, 50, 75, 95)} if v else {}),
                                            "min": float(min(v)) if v else None, "max": float(max(v)) if v else None,
                                            "n_planA_smaller": int(sum(x < 0 for x in v))} for q, v in diffs.items()}
    # 表示
    for key in sorted(k for k in res["source_corners"] if not k.startswith("npts")):
        v = np.array(res["source_corners"][key])
        print("source corners", key, "median %.0f [p5 %.0f, p95 %.0f] min %d max %d (n %d)" % (np.median(v), np.percentile(v, 5), np.percentile(v, 95), v.min(), v.max(), len(v)))
    for key in sorted(k for k in res["source_corners"] if k.startswith("npts")):
        print(key, res["source_corners"][key])
    print("best6 by criterion", res["best6_big_fail_by_criterion"])
    print("classes of best6 trials", Counter(t["stsl_class"] for t in table))
    print("classes (fail only) by ref", {r: Counter(t["stsl_class"] for t in table if t["stsl_fail"] and t["condition"].split("__")[1].startswith(r)) for r in ("E2", "E3")})
    print("paired diffs", res["paired_diffs_both_success"])
    json.dump(res, open(os.path.join(ROOT, "results", "close", "r46_close.json"), "w"), ensure_ascii=False, default=float)
    json.dump(res, open("Registration/output/diag/r46/r46_close.json", "w"), ensure_ascii=False, default=float)


if __name__ == "__main__":
    main()
