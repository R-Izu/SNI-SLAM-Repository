"""R43 の採点と判定。**結果を見る前に commit する。**

対照（判定）：R42 §6-3 のまま。
- 入力 F_10・F_25・F_50 ごとに、同じ 50 試行（同じ摂動）で、案A（label_filter on）と案A（off）の旧基準の合否
  （形 C の d_Ω(T̂P, G, Ω)、回転 < 5°・d_Ω < 0.1 m・縮尺 < 0.05、縮退は不合格。Ω は R40 §4 と同じ err_zero の build_omega）
- n10 = on だけ成功、n01 = off だけ成功。正確 McNemar（両側）：n = n10 + n01、p = min(1, 2·P[Bin(n, 1/2) ≤ min(n10, n01)])
- 支持：3 水準のうち 2 水準以上で n10 − n01 ≥ 10 かつ p < 0.05
  支持されない：3 水準すべてで |n10 − n01| ≤ 3
  判断保留：それ以外
- 各側の成功割合と Wilson 95% CI は記述（判定には使わない）。誤差なし（F_0）も記述

実データ（記述のみ。R41 と同じ量）：廊下 8 条件 × seed 1〜10、on / off
- G1〜G8 ごとに、同時到達割合 F(a = 5°, b ∈ {0.3, 1} m, c = 0.05)、向きの取り違え（e_R ≥ 45°）
- 対応比較 off − on の Δe_Ω・Δe_R と、向きが G1〜G6、G1〜G8 で保たれるか
"""
import json, os, sys
from math import comb
import numpy as np
import yaml

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "Registration")); sys.path.insert(0, os.path.join(REPO, "Registration", "scripts"))
os.chdir(REPO)
from criterion_verdict import build_omega, d_omega                     # noqa: E402
from failure_decomposition import provenance                           # noqa: E402
from r37_summary import wilson                                         # noqa: E402
from r41_rescore import per_trial, real_criteria, F                    # noqa: E402
from regbim import metrics                                             # noqa: E402

LEVELS = ["F_10", "F_25", "F_50"]
TH = (5.0, 0.1, 0.05)


def mcnemar_exact(n10, n01):
    n = n10 + n01
    if n == 0:
        return 1.0
    k = min(n10, n01)
    return min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) / 2.0 ** n)


def control():
    om = build_omega(yaml.safe_load(open("Registration/configs/r40_bim/err_zero.yaml")))
    out = {}
    for name in ["F_0"] + LEVELS:
        side = {}
        for lf in ("on", "off"):
            tm = json.load(open("output/Registration/r43_bim/%s_%s/trial_matrices.json" % (name, lf)))
            G = np.asarray(json.load(open(tm["T_gt"]["path"]))["T_gt"], dtype=np.float64).reshape(4, 4)
            rows = {}
            for t in tm["trials"]:
                if t["trial"] < 0 or t["method"] != "proposed":
                    continue
                p = per_trial(t["T_est"], t["P"], G, om, None)
                p["ok"] = bool(p["v"] and p["eR"] < TH[0] and p["eO"] < TH[1] and p["q"] < TH[2])
                rows[t["trial"]] = p
            side[lf] = rows
        trials = sorted(side["on"])
        assert trials == sorted(side["off"]) and len(trials) == 50, (name, len(trials))
        on = [side["on"][i]["ok"] for i in trials]; off = [side["off"][i]["ok"] for i in trials]
        n10 = sum(a and not b for a, b in zip(on, off)); n01 = sum(b and not a for a, b in zip(on, off))
        ent = {"n_on": sum(on), "n_off": sum(off), "n10": n10, "n01": n01, "p_mcnemar": mcnemar_exact(n10, n01),
               "wilson_on": list(wilson(sum(on), 50)), "wilson_off": list(wilson(sum(off), 50))}
        for lf in ("on", "off"):
            v = list(side[lf].values())
            fin = lambda k: [x[k] for x in v if x["v"] and np.isfinite(x[k])]
            ent["dist_" + lf] = {k: {"p50": float(np.median(fin(k))), "p95": float(np.percentile(fin(k), 95)),
                                     "max": float(np.max(fin(k)))} for k in ("eR", "eO", "q")}
            ent["orient_" + lf] = {o: sum(1 for x in v if x["o"] == o) for o in ("ok", "90", "180", "invalid")}
            ent["fail_" + lf] = {"rot": sum(1 for x in v if not x["ok"] and x["eR"] >= TH[0]),
                                 "trans": sum(1 for x in v if not x["ok"] and x["eO"] >= TH[1]),
                                 "scale": sum(1 for x in v if not x["ok"] and x["q"] >= TH[2])}
        ent["trials"] = {lf: {str(i): side[lf][i] for i in trials} for lf in ("on", "off")}
        out[name] = ent
    d = [out[n]["n10"] - out[n]["n01"] for n in LEVELS]
    sup = sum(1 for n in LEVELS if out[n]["n10"] - out[n]["n01"] >= 10 and out[n]["p_mcnemar"] < 0.05)
    if sup >= 2:
        v = "支持"
    elif all(abs(x) <= 3 for x in d):
        v = "支持されない"
    else:
        v = "判断保留"
    return out, {"diff_n10_minus_n01": dict(zip(LEVELS, d)), "n_levels_supporting": sup, "verdict": v}


def real():
    rr = json.load(open("Registration/output/diag/r43/r43_real.json"))
    om_all = np.load("Registration/output/diag/omega.npz"); omb_all = np.load("Registration/output/diag/r39_omega_bim.npz")
    cache, rows = {}, []
    for r in rr["rows"]:
        scene = r["target"].split("__")[0]
        if scene not in cache:
            cache[scene] = real_criteria(scene)
        omb = omb_all[r["target"]] if r["target"] in omb_all.files else None
        per = {lf: {j: per_trial(r["T_" + lf], None, G, om_all[scene], omb) for j, G in cache[scene].items()}
               for lf in ("on", "off")}
        rows.append({"target": r["target"], "k": r["k"], "per": per})
    out = {"n": len(rows), "on_eq_r37_S_off": rr["n_on_eq_r37"], "per_criterion": {}}
    for j in range(1, 9):
        e = {}
        for lf in ("on", "off"):
            tr = [x["per"][lf][j] for x in rows]
            e[lf] = {"F_5_0.3": F(tr, 5, 0.3), "F_5_1": F(tr, 5, 1.0),
                     "F_BIM_5_0.3": F(tr, 5, 0.3, key="eOB"), "F_BIM_5_1": F(tr, 5, 1.0, key="eOB"),
                     "orient": {o: sum(1 for x in tr if x["o"] == o) for o in ("ok", "90", "180", "invalid")},
                     "eR_p50": float(np.median([x["eR"] for x in tr])), "eO_p50": float(np.median([x["eO"] for x in tr]))}
        dO = [x["per"]["off"][j]["eO"] - x["per"]["on"][j]["eO"] for x in rows]
        dR = [x["per"]["off"][j]["eR"] - x["per"]["on"][j]["eR"] for x in rows]
        e["d_eO"] = {"p50": float(np.median(dO)), "min": float(np.min(dO)), "max": float(np.max(dO)),
                     "n_pos": int(sum(v > 0 for v in dO)), "n_neg": int(sum(v < 0 for v in dO)), "n_zero": int(sum(v == 0 for v in dO))}
        e["d_eR"] = {"p50": float(np.median(dR)), "n_pos": int(sum(v > 0 for v in dR)), "n_neg": int(sum(v < 0 for v in dR)),
                     "n_zero": int(sum(v == 0 for v in dR))}
        out["per_criterion"][j] = e

    def consistent(key, js):
        n = 0
        for x in rows:
            s = {np.sign(x["per"]["off"][j][key] - x["per"]["on"][j][key]) for j in js}
            n += len(s) == 1
        return n
    out["consistent"] = {"eO_1_6": consistent("eO", range(1, 7)), "eO_1_8": consistent("eO", range(1, 9)),
                         "eR_1_6": consistent("eR", range(1, 7)), "eR_1_8": consistent("eR", range(1, 9))}
    out["n_identical_T"] = sum(1 for r in rr["rows"] if np.array_equal(np.asarray(r["T_on"]), np.asarray(r["T_off"])))
    out["rows"] = [{"target": x["target"], "k": x["k"],
                    "G1": {lf: {k: x["per"][lf][1][k] for k in ("eR", "eO", "eOB", "q", "o")} for lf in ("on", "off")}}
                   for x in rows]
    return out


def main():
    ctl, judg = control()
    res = {"provenance": provenance(), "control": ctl, "judgment": judg}
    if os.path.exists("Registration/output/diag/r43/r43_real.json"):
        res["real"] = real()
    json.dump(res, open("Registration/output/diag/r43/r43_score.json", "w"), ensure_ascii=False, default=float)
    for n in ["F_0"] + LEVELS:
        c = ctl[n]
        print(n, "on %d off %d n10 %d n01 %d p %.4g" % (c["n_on"], c["n_off"], c["n10"], c["n01"], c["p_mcnemar"]))
    print(judg)


if __name__ == "__main__":
    main()
