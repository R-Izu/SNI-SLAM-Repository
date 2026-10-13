"""R46 段階1の採点（sni-slam の環境）。**回す前に commit する。** 判定の言葉は出さない（S6）。

条件：8 target（cor × E2・E3。E2 の参照は 411、E3 は 410＋411）× 切り出し r ∈ {0.5, 3, 8, ∞} = 32、各 10 試行。
方式：planA・planA_fixed・centroid（SNI-SLAM、`results_sni/`）、STSL（設定 A・B × K = 5、`results/real_<target>_r<r>/`）。

試行ごとの量（R41 §3-1 と同じ。`r41_rescore.per_trial`、基準 G1〜G8、Ω・Ω_BIM は R41 と同じ保存済みのもの）
- SNI-SLAM：T_est と P から形 C（T̂P と G）
- STSL：source → BIM の変換 = Shift(+z_floor) · tf · T_rp · A · Rg（入力 q = A Rg p）。P = I として採点
  **候補が 0（または最適スコアが有限でない）なら無効な出力（v = 0）**（§9-2）
**大きな失敗**（§3-3、固定）：v = 0、または e_R ≥ 45°、または e_Ω ≥ 1.0 m（G1）。G1〜G8 それぞれでも数える
STSL の試行の失敗（§8-5）：5 回のうち 3 回以上が大きな失敗なら、その試行は失敗（多数決）。5 回のうちの成功の割合も出す
STSL の設定（§8-3）：条件ごとに、大きな失敗（多数決）が少ない方の設定を採る（同数なら A）。両方の設定の数も出す
対の比較：同じ条件・同じ試行で、planA と STSL（採った設定）の片方だけ失敗 n10（planA だけ失敗）・n01（STSL だけ失敗）、
正確 McNemar（両側）。planA_fixed と STSL、planA と centroid も同じ形で出す。件数に Wilson 95%
STSL 自身の判定（2D の並進 3 m・ヨー 5°）の件数と、候補なし（単位行列）の件数も並べる
同時到達割合（R41 の格子、G1）を、方式 × 切り出しの段ごと（8 target を合わせて 80 試行）に出す
失敗の分類：STSL は「候補なし／候補に正解に近いものがあったが別を選んだ／候補に正解に近いものが無い」（STSL 自身の判定の
正否で数える）。案A は別の診断（R32 §3）で分ける
時間：STSL は前処理（平面分割・Python）、BIM の索引作成（初回の BIM 読み込み）、登録。SNI-SLAM は 1 試行の登録全体
"""
import glob, json, os, re, sys
from math import isfinite
import numpy as np

REPO = os.path.expanduser("~/rizu/SNI-SLAM")
sys.path.insert(0, os.path.join(REPO, "Registration")); sys.path.insert(0, os.path.join(REPO, "Registration", "scripts"))
os.chdir(REPO)
from failure_decomposition import provenance                     # noqa: E402
from r37_summary import wilson                                   # noqa: E402
from r41_rescore import per_trial, real_criteria, curve, dist     # noqa: E402
from scipy.special import comb as _comb                          # noqa: E402

ROOT = os.environ.get("R46_STSL_ROOT", os.path.expanduser("~/rizu/r46_stsl_data"))
TARGETS = ["m3_cor_%s__%s" % (s, e) for s in "abcd" for e in ("E2", "E3")]
CUTS = ["0p5", "3", "8", "inf"]
SNI = ["planA", "planA_fixed", "centroid"]
FLOOR = {"E2": "M3_411", "E3": "M3_410_411"}
K = 5


def jl(line):
    return json.loads(re.sub(r"(?<![A-Za-z])(-?)inf(?![A-Za-z])", lambda m: m.group(1) + "Infinity", line).replace("nan", "NaN"))


def mcnemar(n10, n01):
    n = n10 + n01
    if n == 0:
        return 1.0
    return min(1.0, 2.0 * sum(_comb(n, i, exact=True) for i in range(min(n10, n01) + 1)) / 2.0 ** n)


def big(p):
    return (not p["v"]) or p["eR"] >= 45.0 or p["eO"] >= 1.0


def main():
    om_all = np.load("Registration/output/diag/omega.npz"); omb_all = np.load("Registration/output/diag/r39_omega_bim.npz")
    res = {"provenance": provenance(), "conditions": {}}
    pooled = {cut: {m: [] for m in SNI + ["STSL_A", "STSL_B", "STSL_best"]} for cut in CUTS}
    for target in TARGETS:
        scene, ref = target.split("__")
        Gs = real_criteria(scene); om = om_all[scene]; omb = omb_all[target]
        zf = json.load(open(os.path.join(ROOT, "BIM", FLOOR[ref], "meta.json")))["z_floor"]
        S = np.eye(4); S[2, 3] = zf
        for cut in CUTS:
            name = "%s_r%s" % (target, cut)
            ent = {"methods": {}}
            sni = json.load(open(os.path.join(ROOT, "results_sni", name + ".json")))["rows"]
            for m in SNI:
                rows = []
                for k in range(10):
                    r = sni["%s|%d" % (m, k)]
                    per = {j: per_trial(r["T_est"], r["P"], G, om, omb) for j, G in Gs.items()}
                    rows.append({"per": per, "time_s": r["time_s"]})
                ent["methods"][m] = rows
            spec = json.load(open(os.path.join(ROOT, "inputs", "real_" + name, "spec.json")))
            for st in "AB":
                runs = {i: [] for i in range(10)}
                for k in range(1, K + 1):
                    seq = "real_%s_%s_k%d" % (name, st, k)
                    trp = json.load(open(os.path.join(ROOT, "processed", seq, "30m", "T_rp.json")))
                    lines = [jl(l) for l in open(os.path.join(ROOT, "results", "real_" + name, "%s_k%d_%s.jsonl" % (st, k, FLOOR[ref])))]
                    pt = json.load(open(os.path.join(ROOT, "results", "real_" + name, "%s_k%d_prep_time.json" % (st, k))))
                    for r in lines:
                        i = r["index"]; t = spec["trials"][i]
                        T = S @ np.asarray(r["tf"]).reshape(4, 4) @ np.asarray(trp[str(i)]) @ np.asarray(t["A"]) @ np.asarray(t["Rg"])
                        valid = len(r["candidates"]) > 0 and isfinite(r["optimal_score"])
                        per = {j: per_trial(T, None, G, om, omb) for j, G in Gs.items()}
                        if not valid:
                            for j in per:
                                per[j] = dict(per[j], v=False)
                        cand = r["candidates"]
                        cls = ("prep_failed" if i in pt.get("prep_failed", []) else "no_candidate" if not cand else
                               ("chosen_correct" if r["success_stsl"] else
                                ("correct_existed_other_chosen" if any(c["correct"] for c in cand) else "no_correct_candidate")))
                        runs[i].append({"per": per, "success_stsl": bool(r["success_stsl"]) and valid, "no_candidate": not cand,
                                        "class": cls, "reg_s": r["reg_s"], "bim_load_s": r["bim_load_s"],
                                        "prep_s_per_input": (pt["plane_seg_s"] + pt["python_prep_s"]) / pt["n_inputs"]})
                ent["methods"]["STSL_" + st] = [{"runs": runs[i]} for i in range(10)]
            # 大きな失敗（G1 と各基準）
            def fails(m, j):
                if m.startswith("STSL"):
                    return [sum(big(x["per"][j]) for x in tr["runs"]) >= 3 for tr in ent["methods"][m]]
                return [big(tr["per"][j]) for tr in ent["methods"][m]]
            fA, fB = sum(fails("STSL_A", 1)), sum(fails("STSL_B", 1))
            best = "STSL_A" if fA <= fB else "STSL_B"
            ent["methods"]["STSL_best"] = ent["methods"][best]; ent["stsl_best_setting"] = best[-1]
            summ = {}
            for m in SNI + ["STSL_A", "STSL_B", "STSL_best"]:
                f1 = fails(m, 1)
                summ[m] = {"n_big_fail": sum(f1), "wilson_fail": list(wilson(sum(f1), 10)),
                           "n_big_fail_by_criterion": {j: sum(fails(m, j)) for j in range(1, 9)}}
                if m.startswith("STSL"):
                    tr = ent["methods"][m]
                    summ[m].update({"frac_runs_ok": [sum(not big(x["per"][1]) for x in t["runs"]) / K for t in tr],
                                    "n_success_stsl_majority": sum(sum(x["success_stsl"] for x in t["runs"]) >= 3 for t in tr),
                                    "n_no_candidate_majority": sum(sum(x["no_candidate"] for x in t["runs"]) >= 3 for t in tr),
                                    "classes": {c: sum(x["class"] == c for t in tr for x in t["runs"]) for c in
                                                ("chosen_correct", "prep_failed", "no_candidate", "correct_existed_other_chosen", "no_correct_candidate")}})
            pairs = {}
            for a, b in (("planA", "STSL_best"), ("planA_fixed", "STSL_best"), ("planA", "centroid"), ("planA", "planA_fixed")):
                fa, fb = fails(a, 1), fails(b, 1)
                n10 = sum(x and not y for x, y in zip(fa, fb)); n01 = sum(y and not x for x, y in zip(fa, fb))
                pairs["%s_vs_%s" % (a, b)] = {"n10_first_only_fails": n10, "n01_second_only_fails": n01, "p_mcnemar": mcnemar(n10, n01)}
            ent["summary"] = summ; ent["pairs"] = pairs
            for m in SNI:
                pooled[cut][m] += [tr["per"][1] for tr in ent["methods"][m]]
            for m in ("STSL_A", "STSL_B", "STSL_best"):
                pooled[cut][m] += [x["per"][1] for tr in ent["methods"][m] for x in tr["runs"]]
            res["conditions"][name] = {"summary": summ, "pairs": pairs, "stsl_best_setting": ent["stsl_best_setting"],
                                       "trials": {m: (ent["methods"][m] if m in SNI else ent["methods"][m]) for m in SNI + ["STSL_A", "STSL_B"]}}
            print(name, {m: summ[m]["n_big_fail"] for m in summ}, "best", ent["stsl_best_setting"],
                  "| planA vs STSL", pairs["planA_vs_STSL_best"], flush=True)
    res["curves_G1"] = {cut: {m: curve(v, "eO") for m, v in d.items()} for cut, d in pooled.items()}
    res["dist_G1"] = {cut: {m: {q: dist(v, q) for q in ("eR", "eO", "eOB", "q")} for m, v in d.items()} for cut, d in pooled.items()}
    os.makedirs("Registration/output/diag/r46", exist_ok=True)
    json.dump(res, open("Registration/output/diag/r46/r46_score.json", "w"), ensure_ascii=False, default=float)
    print("wrote Registration/output/diag/r46/r46_score.json")


if __name__ == "__main__":
    main()
