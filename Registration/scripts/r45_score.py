"""R45 の採点（R41 の仕様）。**回す前に commit する。** 判定は置かない（R45 §1）。

入力：`output/Registration/r45/{plan_a,centroid}/<scene>/trial_matrices.json`（どちらも reference.seed 0、同じ commit）と
R27 の `output/Registration/gt_a/<scene>/trial_matrices.json`（proposed、seed なしの参照、記述用）。
正解 G は各 trial_matrices の T_gt、Ω は R38・R41 と同じ `build_omega(gt_a/<scene>.yaml, seed=0)`。

出すもの（R45 §4・§6）
1. 旧基準の件数（形 C、回転 < 5°・d_Ω < 0.1 m・縮尺 < 0.05、無効は不合格）：案A・centroid・R27、シーン別と全体、Wilson 95%
2. 案A と centroid の対：n10（案A だけ成功）・n01（centroid だけ成功）、Δ = 案A − centroid の e_R・e_Ω・q の分布
3. 分布（e_R・傾き・ヨー・e_Ω・q の 5/25/50/75/95%・最小・最大、無効の件数）、向きの取り違え、同時到達割合の曲線（R41 の格子）
4. 時間（trials.csv の time_s）：中央・最大、300 s 超の件数
5. 失敗した試行の一覧と 2 項分解（R35：重心の差と、重心まわりの回転・縮尺）
6. seed 0 の centroid と R27 の比較：シーンごとの件数、違った試行、分布の比較、行列の差 max|ΔT|
7. 摂動 P が R27 と一致するか（全 800 × 2 方式）
"""
import csv, json, os, sys
import numpy as np
import yaml

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "Registration")); sys.path.insert(0, os.path.join(REPO, "Registration", "scripts"))
os.chdir(REPO)
from criterion_verdict import build_omega                               # noqa: E402
from failure_decomposition import provenance                            # noqa: E402
from r35_rescore import decompose                                       # noqa: E402
from r37_summary import wilson                                          # noqa: E402
from r41_rescore import per_trial, curve, dist, A_GRID, B_GRID          # noqa: E402

SCENES = ["room_0", "room_1", "room_2", "office_0", "office_1", "office_2", "office_3", "office_4"]
OUT = "Registration/output/diag/r45"
TH = (5.0, 0.1, 0.05)


def ok(p):
    return bool(p["v"] and p["eR"] < TH[0] and p["eO"] < TH[1] and p["q"] < TH[2])


def load(path):
    tm = json.load(open(path))
    G = np.asarray(json.load(open(tm["T_gt"]["path"]))["T_gt"], dtype=np.float64).reshape(4, 4)
    return G, {t["trial"]: t for t in tm["trials"] if t["method"] == "proposed" and t["trial"] >= 0}


def times(path):
    return [float(x["time_s"]) for x in csv.DictReader(open(path)) if x["method"] == "proposed" and int(x["trial"]) >= 0]


def main():
    os.makedirs(OUT, exist_ok=True)
    res = {"provenance": provenance(), "thresholds": TH, "scenes": {}}
    allrows = {"plan_a": [], "centroid": [], "r27": []}
    for s in SCENES:
        om = build_omega(yaml.safe_load(open("Registration/configs/gt_a/%s.yaml" % s)), seed=0)
        G27, r27 = load("output/Registration/gt_a/%s/trial_matrices.json" % s)
        runs = {"r27": (G27, r27)}
        for m in ("plan_a", "centroid"):
            runs[m] = load("output/Registration/r45/%s/%s/trial_matrices.json" % (m, s))
        trials = sorted(r27)
        assert all(sorted(runs[m][1]) == trials for m in runs) and len(trials) == 100, s
        per = {m: {k: per_trial(runs[m][1][k]["T_est"], runs[m][1][k]["P"], runs[m][0], om, None) for k in trials} for m in runs}
        for m in per:
            for k in trials:
                per[m][k]["ok"] = ok(per[m][k])
                allrows[m].append(per[m][k])
        ent = {"G_equal_r27": {m: bool(np.array_equal(runs[m][0], G27)) for m in ("plan_a", "centroid")},
               "P_equal_r27": {m: int(sum(np.array_equal(np.asarray(runs[m][1][k]["P"]), np.asarray(r27[k]["P"])) for k in trials))
                               for m in ("plan_a", "centroid")}}
        for m in per:
            k_ok = sum(per[m][k]["ok"] for k in trials)
            ent[m] = {"n_success": k_ok, "N": 100, "wilson": list(wilson(k_ok, 100)),
                      "dist": {q: dist(list(per[m].values()), q) for q in ("eR", "tilt", "yaw", "eO", "q")},
                      "orientation": {o: sum(1 for k in trials if per[m][k]["o"] == o) for o in ("ok", "90", "180", "invalid")},
                      "curve_eO": curve(list(per[m].values()), "eO")}
        for m in ("plan_a", "centroid"):
            t = times("output/Registration/r45/%s/%s/trials.csv" % (m, s))
            ent[m]["time"] = {"median_s": float(np.median(t)), "max_s": float(np.max(t)), "n_over_300s": int(sum(x > 300 for x in t))}
        # 対：案A − centroid
        a, c = per["plan_a"], per["centroid"]
        ent["pair"] = {"n10": sum(a[k]["ok"] and not c[k]["ok"] for k in trials),
                       "n01": sum(c[k]["ok"] and not a[k]["ok"] for k in trials),
                       "delta": {q: dist([{"v": True, q: a[k][q] - c[k][q]} for k in trials
                                          if a[k]["v"] and c[k]["v"]], q) for q in ("eR", "eO", "q")}}
        # seed 0 の centroid と R27
        ent["centroid_vs_r27"] = {
            "differing_trials": [{"trial": k, "centroid_ok": c[k]["ok"], "r27_ok": per["r27"][k]["ok"],
                                  "eR": [c[k]["eR"], per["r27"][k]["eR"]], "eO": [c[k]["eO"], per["r27"][k]["eO"]]}
                                 for k in trials if c[k]["ok"] != per["r27"][k]["ok"]],
            "max_abs_dT": float(max(np.abs(np.asarray(runs["centroid"][1][k]["T_est"]) - np.asarray(r27[k]["T_est"])).max() for k in trials)),
            "delta_eO_median": float(np.median([c[k]["eO"] - per["r27"][k]["eO"] for k in trials]))}
        # 失敗の一覧と 2 項分解
        ent["failures"] = {}
        for m in ("plan_a", "centroid"):
            f = []
            for k in trials:
                if not per[m][k]["ok"]:
                    Tc = np.asarray(runs[m][1][k]["T_est"]) @ np.asarray(runs[m][1][k]["P"])
                    dec = decompose(Tc, runs[m][0], om) if per[m][k]["v"] else None
                    f.append({"trial": k, **{q: per[m][k][q] for q in ("eR", "tilt", "yaw", "eO", "q", "o")}, "decomp": dec})
            ent["failures"][m] = f
        ent["trials"] = {m: {str(k): per[m][k] for k in trials} for m in per}
        res["scenes"][s] = ent
        print(s, "plan_a %d centroid %d r27 %d | n10 %d n01 %d | P_eq %s | cen vs r27 diff %d max|dT| %.4f"
              % (ent["plan_a"]["n_success"], ent["centroid"]["n_success"], ent["r27"]["n_success"], ent["pair"]["n10"],
                 ent["pair"]["n01"], ent["P_equal_r27"], len(ent["centroid_vs_r27"]["differing_trials"]),
                 ent["centroid_vs_r27"]["max_abs_dT"]), flush=True)
    tot = {}
    for m in allrows:
        k = sum(r["ok"] for r in allrows[m]); n = len(allrows[m])
        tot[m] = {"n_success": k, "N": n, "wilson": list(wilson(k, n)),
                  "dist": {q: dist(allrows[m], q) for q in ("eR", "tilt", "yaw", "eO", "q")},
                  "orientation": {o: sum(1 for r in allrows[m] if r["o"] == o) for o in ("ok", "90", "180", "invalid")},
                  "curve_eO_equal_scene_weight": {"%g" % a: list(np.mean([res["scenes"][s][m]["curve_eO"]["%g" % a] for s in SCENES], axis=0))
                                                  for a in A_GRID}}
    tot["pair"] = {"n10": sum(res["scenes"][s]["pair"]["n10"] for s in SCENES), "n01": sum(res["scenes"][s]["pair"]["n01"] for s in SCENES)}
    res["total"] = tot
    res["grid"] = {"a": A_GRID, "b": B_GRID.tolist(), "c": TH[2]}
    json.dump(res, open(os.path.join(OUT, "r45_score.json"), "w"), ensure_ascii=False, default=float)
    print("total", {m: (tot[m]["n_success"], tot[m]["N"]) for m in allrows}, tot["pair"])


if __name__ == "__main__":
    main()
