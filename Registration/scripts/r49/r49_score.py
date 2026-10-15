"""R49 段階1 の採点（sni-slam の環境）。**回す前に commit する。** 判定の言葉は出さない（S4）。指示書 §2・§8。

試行の採点（R45〜R48 と同じ）
- `r41_rescore.per_trial(T_est, P, G, Ω)`。大きな失敗＝無効・e_R ≥ 45°・e_Ω ≥ 1.0 m
- synth：G = R45 の T_gt、Ω = `build_omega(gt_a/<scene>.yaml, seed=0)`。bim：G = R39 の T_gt、Ω = `build_omega(r39_bim の config)`。
  real：G1（`real_criteria`）、Ω = `omega.npz`。G2〜G8 は感度として失敗の数だけ出す
- 段ごとの失敗の分類（R32 §3、`r46_planA_diag.py`・`r47_planA_diag.py` と同じ規則。案A 系で、診断を記録した試行）：
  E_gen・E_short（相関の候補・短い精緻化の後の最小の d_Ω）、E_after_icp（通常の精緻化の後の候補の最小）、最終。
  「良い候補が残ったが別が選ばれた」（E_short か E_after_icp が最終より 0.15 m 以上小さい）／「精緻化で悪化」（最終 > E_gen + 0.15）／
  「精緻化がほぼ動かしていない」（|最終 − E_gen| < 0.15）／「精緻化で改善」
主な検定（水準 0.05/4 = 0.0125）
- T1：synth、完全 対 壁の向きからのヨーの候補なし の大きな失敗、試行の対（800）の正確 McNemar、両側
- T2：real（G1）、同上（80 対）
- T3：synth、完全 対 クラス別なし、両方式が大きな失敗をしなかった対で、**勝った候補の通常の精緻化 1 回の反復の数**
  （判定 ‖ΔT‖_F < 1e-5 で止まった番号、上限なら 50）の Wilcoxon 符号順位（両側、scipy の既定）
- T4：同じ対で、**1 試行の ICP の呼び出しすべての時間の和**（KD 木を作る時間を含む）の Wilcoxon 符号順位（両側）
副・記述
- T3 の副：§2-3 の後半の定義（反復の変化が回転 0.01° 未満・並進 1 mm 未満・縮尺 1e-4 未満になった最初の番号、無ければ 50）
- 反復の範囲 (b) 通常の精緻化 4 回の和、(c) すべての呼び出しの和。短い精緻化が上限で止まった割合。上限で止まった数
- T4 の副：段2 の精緻化（結果を使わない）を除いた和
- シーンごとの中央値の差（8 シーン）と向きがそろっているか。勝った候補の番号が 2 方式で違う対の数と、それを除いた T3（事後）
- 完全な方式：R45 の T_est とビット一致しない試行の数、大きな失敗の判定が R45 と変わった試行の数（変わった場合、T1 は今回の値を使う）
- 方式 × データの大きな失敗・誤差の分布・段の分類、e_R の分布（壁の向きからのヨーの候補なし と完全）、段ごとの時間（完全、3 データ）
出力：`Registration/output/diag/r49/r49_score.json`
"""
import json, os, sys
from collections import Counter
import numpy as np
import yaml

REPO = os.path.expanduser("~/rizu/SNI-SLAM")
sys.path.insert(0, os.path.join(REPO, "Registration")); sys.path.insert(0, os.path.join(REPO, "Registration", "scripts"))
sys.path.insert(0, os.path.join(REPO, "Registration", "scripts", "r49"))
os.chdir(REPO)
from criterion_verdict import build_omega, d_omega              # noqa: E402
from failure_decomposition import provenance                    # noqa: E402
from r41_rescore import per_trial, real_criteria                # noqa: E402
from scipy.special import comb as _comb                         # noqa: E402
from scipy.stats import wilcoxon                                # noqa: E402
import r49_run as RR                                            # noqa: E402

ALPHA = 0.05 / 4
CAP = 50


def mcnemar(n10, n01):
    n = n10 + n01
    return 1.0 if n == 0 else min(1.0, 2.0 * sum(_comb(n, i, exact=True) for i in range(min(n10, n01) + 1)) / 2.0 ** n)


def big(p):
    return (not p["v"]) or p["eR"] >= 45.0 or p["eO"] >= 1.0


def q(v):
    v = [x for x in v if x is not None and np.isfinite(x)]
    return None if not v else {"n": len(v), "p5": float(np.percentile(v, 5)), "p25": float(np.percentile(v, 25)), "median": float(np.median(v)),
                               "p75": float(np.percentile(v, 75)), "p95": float(np.percentile(v, 95)), "max": float(np.max(v))}


def truth(data, target):
    if data == "synth":
        tm = json.load(open("output/Registration/r45/plan_a/%s/trial_matrices.json" % target))
        G = np.asarray(json.load(open(tm["T_gt"]["path"]))["T_gt"], dtype=np.float64).reshape(4, 4)
        om = build_omega(yaml.safe_load(open("Registration/configs/gt_a/%s.yaml" % target)), seed=0)
        return {"1": G}, om, tm
    if data == "bim":
        cfg = yaml.safe_load(open("Registration/configs/r39_bim/bim_%s_plan_correlate.yaml" % target))
        tm = json.load(open("output/Registration/r39_bim/%s_plan_correlate/trial_matrices.json" % target))
        G = np.asarray(json.load(open(tm["T_gt"]["path"]))["T_gt"], dtype=np.float64).reshape(4, 4)
        return {"1": G}, build_omega(cfg), None
    s = target.split("__")[0]
    return {str(j): G for j, G in real_criteria(s).items()}, np.load("Registration/output/diag/omega.npz")[s], None


def stage_class(r, G, om):
    """R32 §3 の段の分類（`r47_planA_diag.py` と同じ規則）。r は診断（diag）を含む実行の行"""
    d = r.get("diag")
    if not d:
        return None
    P = np.asarray(r["P"], dtype=np.float64)
    dO = lambda X: float(d_omega(np.asarray(X, dtype=np.float64) @ P, G, om))
    gen = min(dO(x) for x in d["gen"]); sho = min(dO(x) for x in d["short"]); aft = min(dO(x) for x in d["after_icp"])
    fin = dO(r["T_est"])
    v = ("良い候補が残ったが別が選ばれた" if (sho < fin - 0.15 or aft < fin - 0.15) else "精緻化で悪化" if fin > gen + 0.15
         else "精緻化がほぼ動かしていない" if abs(fin - gen) < 0.15 else "精緻化で改善")
    return {"E_gen": gen, "E_short": sho, "E_after_icp": aft, "final": fin, "class": v}


def iters_def2(deltas):
    for i, d in enumerate(deltas, 1):
        if d["rot_deg"] < 0.01 and d["trans"] < 1e-3 and d["scale_rel"] < 1e-4:
            return i
    return CAP


def winner_normal(instr):
    normals = [c for c in instr["icp"] if c["tag"] == "icp_normal"]
    return normals[instr["winner_index"]] if normals and instr.get("winner_index") is not None else None


def iter_measures(instr):
    w = winner_normal(instr)
    if w is None:
        return None
    a = w["iters"] if w["stop"] == "conv" else CAP
    return {"a": a, "a_def2": iters_def2(w["deltas"]), "a_hit_cap": w["stop"] == "max",
            "b": sum(c["iters"] for c in instr["icp"] if c["tag"] == "icp_normal"),
            "c": sum(c["iters"] for c in instr["icp"]),
            "short_cap_frac": float(np.mean([c["stop"] == "max" for c in instr["icp"] if c["tag"] == "icp_short"] or [np.nan])),
            "n_normal_cap": sum(c["stop"] == "max" for c in instr["icp"] if c["tag"] == "icp_normal"),
            "t_icp": sum(c["t_total"] for c in instr["icp"]),
            "t_icp_no_refine": sum(c["t_total"] for c in instr["icp"] if c["tag"] != "icp_refine"),
            "winner_index": instr["winner_index"]}


def wilc(x, y):
    d = np.asarray(x, float) - np.asarray(y, float)
    if len(d) == 0 or np.all(d == 0):
        return {"n": int(len(d)), "n_nonzero": 0, "p": 1.0}
    w = wilcoxon(x, y, alternative="two-sided")
    return {"n": int(len(d)), "n_nonzero": int((d != 0).sum()), "n_first_larger": int((d > 0).sum()), "n_first_smaller": int((d < 0).sum()),
            "median_diff": float(np.median(d)), "statistic": float(w.statistic), "p": float(w.pvalue)}


def main():
    res = {"provenance": provenance(), "alpha": ALPHA, "data": {}}
    trial = {}                                       # (data, method, target, k) → 採点
    for data in ("synth", "bim", "real"):
        for target in RR.targets(data):
            Gs, om, tm45 = truth(data, target)
            for m in RR.METHOD:
                p = RR.path(data, m, target)
                if not os.path.exists(p):
                    continue
                rows = json.load(open(p))["rows"]
                for k, r in rows.items():
                    per = {j: per_trial(r["T_est"], r["P"], G, om, None) for j, G in Gs.items()}
                    e = {"per": {j: {kk: per[j][kk] for kk in ("v", "eR", "eO", "q")} for j in per}, "fail": {j: big(per[j]) for j in per},
                         "time_s": r["time_s"], "read_s": r.get("read_s")}
                    if r.get("instr"):
                        e["iter"] = iter_measures(r["instr"]); e["stages"] = r["instr"]["stages"]
                    if e["fail"]["1"] and r.get("diag"):
                        e["stage_class"] = stage_class(r, Gs["1"], om)
                    if data == "synth" and m == "full":
                        r45 = {t["trial"]: t for t in tm45["trials"] if t["method"] == "proposed" and t["trial"] >= 0}[int(k)]
                        p45 = per_trial(r45["T_est"], r45["P"], Gs["1"], om, None)
                        e["r45_bit_equal"] = bool(np.array_equal(np.asarray(r["T_est"]), np.asarray(r45["T_est"])))
                        e["r45_fail"] = big(p45)
                    trial[(data, m, target, int(k))] = e
    # 既存の synth の nograv・4 方式（output/Registration/gt_a、R27 以降の benchmark。記述）
    for target in RR.SYNTH:
        Gs, om, _ = truth("synth", target)
        tm = json.load(open("output/Registration/gt_a/%s/trial_matrices.json" % target))
        name = {"proposed_no_gravity": "nograv", "baseline_open3d": "fpfh_p2p", "baseline_ransac_p2l": "fpfh_p2l",
                "baseline_fgr": "fgr_p2p", "baseline_fgr_p2l": "fgr_p2l"}
        for t in tm["trials"]:
            if t["method"] in name and t["trial"] >= 0:
                p = per_trial(t["T_est"], t["P"], Gs["1"], om, None)
                trial[("synth", name[t["method"]] + "_existing", target, int(t["trial"]))] = {"per": {"1": p}, "fail": {"1": big(p)}}
    # 方式 × データの集計
    summ = {}
    for (data, m, target, k), e in trial.items():
        s = summ.setdefault("%s|%s" % (data, m), {"n": 0, "fail": Counter(), "eR": [], "eO": [], "q": [], "time": [], "per_target_fail": Counter(),
                                                  "stage": Counter()})
        s["n"] += 1
        if e.get("stage_class"):
            s["stage"][e["stage_class"]["class"]] += 1
        for j, f in e["fail"].items():
            s["fail"][j] += int(f)
        s["per_target_fail"][target] += int(e["fail"]["1"])
        p = e["per"]["1"]
        if p["v"]:
            s["eR"].append(p["eR"]); s["eO"].append(p["eO"]); s["q"].append(p["q"])
        if e.get("time_s") is not None:
            s["time"].append(e["time_s"])
    res["summary"] = {k: {"n": v["n"], "n_fail_by_criterion": dict(v["fail"]), "fail_per_target": dict(v["per_target_fail"]),
                          "eR": q(v["eR"]), "eO": q(v["eO"]), "q": q(v["q"]), "time_s": q(v["time"]), "fail_stage_class": dict(v["stage"])}
                      for k, v in summ.items()}
    # T1・T2
    tests = {}
    for name, data, targets, n in (("T1", "synth", RR.SYNTH, 100), ("T2", "real", RR.REAL, 10)):
        a = [trial.get((data, "full", t, k)) for t in targets for k in range(n)]; b = [trial.get((data, "nowall", t, k)) for t in targets for k in range(n)]
        pairs = [(x["fail"]["1"], y["fail"]["1"]) for x, y in zip(a, b) if x and y]
        n10 = sum(x and not y for x, y in pairs); n01 = sum(y and not x for x, y in pairs)
        tests[name] = {"n_pairs": len(pairs), "fail_full": sum(x for x, _ in pairs), "fail_nowall": sum(y for _, y in pairs),
                       "n10_full_only": n10, "n01_nowall_only": n01, "p": mcnemar(n10, n01), "alpha": ALPHA}
    # T3・T4
    pairs = []
    for t in RR.SYNTH:
        for k in range(100):
            x, y = trial.get(("synth", "full", t, k)), trial.get(("synth", "single", t, k))
            if x and y and not x["fail"]["1"] and not y["fail"]["1"] and x.get("iter") and y.get("iter"):
                pairs.append((t, k, x["iter"], y["iter"]))
    tests["T3"] = dict(wilc([x["a"] for _, _, x, _ in pairs], [y["a"] for _, _, _, y in pairs]), alpha=ALPHA,
                       note="first = 完全、second = クラス別なし")
    tests["T4"] = dict(wilc([x["t_icp"] for _, _, x, _ in pairs], [y["t_icp"] for _, _, _, y in pairs]), alpha=ALPHA)
    side = {"T3_def2": wilc([x["a_def2"] for _, _, x, _ in pairs], [y["a_def2"] for _, _, _, y in pairs]),
            "T3_b": wilc([x["b"] for _, _, x, _ in pairs], [y["b"] for _, _, _, y in pairs]),
            "T3_c": wilc([x["c"] for _, _, x, _ in pairs], [y["c"] for _, _, _, y in pairs]),
            "T4_no_refine": wilc([x["t_icp_no_refine"] for _, _, x, _ in pairs], [y["t_icp_no_refine"] for _, _, _, y in pairs])}
    same = [(t, k, x, y) for t, k, x, y in pairs if x["winner_index"] == y["winner_index"]]
    side["posthoc_n_pairs_winner_differs"] = len(pairs) - len(same)
    side["posthoc_T3_same_winner"] = wilc([x["a"] for _, _, x, _ in same], [y["a"] for _, _, _, y in same])
    side["per_scene_median_diff"] = {}
    for t in RR.SYNTH:
        ps = [(x, y) for tt, _, x, y in pairs if tt == t]
        if ps:
            side["per_scene_median_diff"][t] = {"n": len(ps), "a": float(np.median([x["a"] - y["a"] for x, y in ps])),
                                                "t_icp": float(np.median([x["t_icp"] - y["t_icp"] for x, y in ps]))}
    for m in ("full", "single"):
        it = [e["iter"] for (d, mm, _, _), e in trial.items() if d == "synth" and mm == m and e.get("iter")]
        side["iter_dist_" + m] = {"a": q([i["a"] for i in it]), "b": q([i["b"] for i in it]), "c": q([i["c"] for i in it]),
                                  "a_hit_cap": sum(i["a_hit_cap"] for i in it), "n": len(it),
                                  "short_cap_frac_mean": float(np.mean([i["short_cap_frac"] for i in it])) if it else None,
                                  "normal_calls_hit_cap": sum(i["n_normal_cap"] for i in it), "t_icp": q([i["t_icp"] for i in it])}
    for data in ("bim", "real"):
        for m in ("full", "single"):
            it = [e["iter"] for (d, mm, _, _), e in trial.items() if d == data and mm == m and e.get("iter")]
            side["iter_dist_%s_%s" % (data, m)] = {"a": q([i["a"] for i in it]), "b": q([i["b"] for i in it]), "c": q([i["c"] for i in it]),
                                                    "a_hit_cap": sum(i["a_hit_cap"] for i in it), "t_icp": q([i["t_icp"] for i in it])}
    # 完全 対 R45
    f = [e for (d, m, _, _), e in trial.items() if d == "synth" and m == "full"]
    side["full_vs_r45"] = {"n": len(f), "not_bit_equal": sum(not e["r45_bit_equal"] for e in f),
                           "fail_judgement_changed": sum(e["fail"]["1"] != e["r45_fail"] for e in f)}
    # 段ごとの時間（完全）
    st = {}
    for data in ("synth", "bim", "real"):
        es = [e for (d, m, _, _), e in trial.items() if d == data and m == "full" and e.get("stages")]
        st[data] = {k: q([e["stages"].get(k, 0.0) for e in es]) for k in ("prep", "rotation", "correlate", "icp", "select")}
        st[data]["total"] = q([e["time_s"] for e in es]); st[data]["read_per_scene"] = q([e["read_s"] for e in es])
    side["stage_times_full"] = st
    res["tests"] = tests; res["side"] = side
    os.makedirs("Registration/output/diag/r49", exist_ok=True)
    json.dump(res, open("Registration/output/diag/r49/r49_score.json", "w"), ensure_ascii=False, default=float)
    for k, v in tests.items():
        print(k, v)
    print(json.dumps({k: side[k] for k in ("posthoc_n_pairs_winner_differs", "full_vs_r45")}))


if __name__ == "__main__":
    main()
