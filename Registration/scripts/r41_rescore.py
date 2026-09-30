"""R41 — 保存済みの出力を 3 層の評価（(a) 同時到達割合の曲線、(b) 全試行の分布）で採点し直す。**手法は再実行しない。**

**これは、既存の結果を確認した後に設計した記述的な再解析である**（R41 §1）。
仕様は採点を始める前にこのファイルに書いて commit した。

仕様（R41 §3。ここで固定）
--------------------------
試行ごとの量（試行 i・基準 G・評価点集合 Ω）
- v   ：有効な出力か。行列が有限で、`is_degenerate_sim3` でない。**無効も母数に残す**（件数を別に出す）
- e_R ：回転誤差 [°]（`rotation_error_deg`）。傾き・ヨー（R36 §3-1 の分け方）も出す
- e_Ω ：正しい形 C。d_Ω(T̂ P, G, Ω)（摂動が無ければ P = I）
- e_ΩBIM：実データのみ。Ω_BIM は R39 §3 で保存した `r39_omega_bim.npz` をそのまま使う（作り直さない。保存が無い条件は欠測）
- q   ：|ŝ/s_G − 1|
- o   ：e_R ≥ 45° なら取り違え。45 ≤ e_R < 135 を 90° 系、135° 以上を 180° 系
同時到達割合 F(a, b, c) = (1/N) Σ 1[v ∧ e_R < a ∧ e_Ω < b ∧ q < c]
- 不等号は元の採点（`criterion_verdict.py` L140–142、`threshold_curve.py` L81–83、`stats.check_success`）と同じ「<」
- 旧判定 = F_{G1,Ω}(5°, 0.1 m, 0.05)
- 格子：a ∈ {1, 2, 5, 10, 45}°、b ∈ 0.01〜30 m の対数等間隔 61 点、c = 0.05 に固定（断面）
- 実データは e_Ω と e_ΩBIM の曲線を別に出す
分布：N・無効の件数・5/25/50/75/95% 点・最小・最大（有効な試行の有限値）。固定条件の集合（D1・D2）は全点も出す
基準感度（実データ）：G1（主）、G1〜G6、G1〜G8 を分けて。合否は基準ごとに AND（指標ごとに別の基準を選ばない）
対応比較：D2 は同じ条件の案A − 既定、D3 は同じ試行の on − off。各 G_j で差を出し、向きが全基準で保たれるか
区間：Wilson 95% を二項モデルを置ける層ごとに点ごとで（D3：条件×系列×側、D4：シーン×手法、D5：入力×手法）。
      D1・D2 は固定条件の集合なので区間を出さない。曲線上の区間は点ごとで、同時帯ではない

データ（R41 §4。混ぜない）
- D1 実データ・既定 40 条件（`realdata_direct_v3.json`。R30 の 6 基準の採点と同じ出力）
- D2 実データ・R35 の 16 出力（既定・案A × 8 条件）
- D3 実データ・R37 の 160 対（off/on、S/P）
- D4 合成 GT-A（R27 の全手法、8 シーン × 100）。解析的な正解、Ω は元の Ω
- D5 BIM 由来の対照（R39 §5 の E2・E3、R40 §4 の全入力、R40 §3 の E2' は別の行に「交絡」と注記）
"""
from __future__ import annotations

import json, os, sys
from collections import defaultdict
import numpy as np, yaml

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "Registration")); sys.path.insert(0, os.path.join(REPO, "Registration", "scripts"))
if not os.path.isdir("Registration"):
    os.chdir(REPO)
from criterion_verdict import build_omega, d_omega                    # noqa: E402
from failure_decomposition import provenance                          # noqa: E402
from r36_rotation_diag import tilt_yaw                                # noqa: E402
from r37_summary import wilson                                        # noqa: E402
from regbim import metrics                                            # noqa: E402

A_GRID = [1.0, 2.0, 5.0, 10.0, 45.0]
B_GRID = np.logspace(np.log10(0.01), np.log10(30.0), 61)
C_FIX = 0.05
OLD = (5.0, 0.1, 0.05)
REP_A, REP_B = [5.0, 45.0], [0.1, 0.3, 1.0, 3.0]
OUT = "Registration/output/diag/r41"
PNG = None


# --------------------------------------------------------------------------- #
def per_trial(T, P, G, om, omb):
    T = np.asarray(T, dtype=np.float64); P = np.eye(4) if P is None else np.asarray(P, dtype=np.float64)
    Tc = T @ P
    if not np.isfinite(Tc).all() or metrics.is_degenerate_sim3(Tc):
        return {"v": False, "eR": np.inf, "tilt": np.inf, "yaw": np.inf, "eO": np.inf, "eOB": (np.inf if omb is not None else None),
                "q": np.inf, "o": "invalid"}
    R, _, s = metrics.decompose_sim3(Tc); RG, _, sG = metrics.decompose_sim3(G)
    eR = metrics.rotation_error_deg(R, RG); ty = tilt_yaw(R, RG)
    o = "ok" if eR < 45 else ("90" if eR < 135 else "180")
    return {"v": True, "eR": eR, "tilt": ty["tilt_deg"], "yaw": ty["yaw_deg"], "eO": d_omega(Tc, G, om),
            "eOB": (d_omega(Tc, G, omb) if omb is not None else None), "q": float(abs(s / sG - 1)), "o": o}


def F(rows, a, b, c=C_FIX, key="eO"):
    if not rows:
        return None
    k = sum(1 for r in rows if r["v"] and r[key] is not None and r["eR"] < a and r[key] < b and r["q"] < c)
    return k / len(rows)


def curve(rows, key="eO"):
    if key == "eOB" and any(r["eOB"] is None for r in rows):
        return None
    return {"%g" % a: [F(rows, a, b, key=key) for b in B_GRID] for a in A_GRID}


def dist(rows, key):
    vals = [r[key] for r in rows if r["v"] and r[key] is not None and np.isfinite(r[key])]
    out = {"N": len(rows), "n_invalid": sum(1 for r in rows if not r["v"])}
    if vals:
        a = np.asarray(vals)
        out.update({"p%d" % p: float(np.percentile(a, p)) for p in (5, 25, 50, 75, 95)})
        out.update({"min": float(a.min()), "max": float(a.max())})
    return out


def old_count(rows, key="eO"):
    return sum(1 for r in rows if r["v"] and r[key] is not None and r["eR"] < OLD[0] and r[key] < OLD[1] and r["q"] < OLD[2])


def wil(rows, a, b, key="eO"):
    n = len(rows); k = round((F(rows, a, b, key=key) or 0) * n)
    return [k, n, list(wilson(k, n))]


# --------------------------------------------------------------------------- #
def real_criteria(scene):
    sp = json.load(open("Registration/output/diag/criterion_spread.json"))["scenes"][scene]
    Gs = {int(f["id"]): np.asarray(f["T"], dtype=np.float64) for f in sp["results"] if f.get("T") is not None}
    assert sorted(Gs) == [1, 2, 3, 4, 5, 6], (scene, sorted(Gs))
    c = json.load(open("Registration/output/diag/r39_criteria.json"))["scenes"][scene]["T"]
    Gs[7], Gs[8] = np.asarray(c["7"]), np.asarray(c["8"])
    return Gs


def load_real():
    om_all = np.load("Registration/output/diag/omega.npz"); omb_all = np.load("Registration/output/diag/r39_omega_bim.npz")
    items = []   # (dataset, stratum, method, series, trial, T, P, scene, target)
    for r in json.load(open("Registration/output/diag/realdata_direct_v3.json"))["results"]:
        items.append(("D1", r["target"], "default", None, 0, r["T_est"], None, r["scene"], r["target"]))
    for r in json.load(open("Registration/output/diag/r35_runs.json"))["rows"]:
        t = "%s__%s" % (r["scene"], r["cond"])
        items.append(("D2", t, "planA" if r["mode"] == "plan_correlate" else "default", None, 0, r["T_est"], None, r["scene"], t))
    for r in json.load(open("Registration/output/diag/r37_replicate.json"))["rows"]:
        for side in ("off", "on"):
            items.append(("D3", r["target"], side, r["series"], r["k"], r["T_" + side], r["P"], r["target"].split("__")[0], r["target"]))
    rows = []
    cache = {}
    for ds, st, m, ser, k, T, P, scene, target in items:
        if scene not in cache:
            cache[scene] = real_criteria(scene)
        omb = omb_all[target] if target in omb_all.files else None
        per = {j: per_trial(T, P, G, om_all[scene], omb) for j, G in cache[scene].items()}
        rows.append({"dataset": ds, "stratum": st, "method": m, "series": ser, "trial": k, "scene": scene, "per": per})
    return rows


def load_synth():
    rows = []
    # D4
    for s in ["room_0", "room_1", "room_2", "office_0", "office_1", "office_2", "office_3", "office_4"]:
        tm = json.load(open("output/Registration/gt_a/%s/trial_matrices.json" % s))
        G = np.asarray(json.load(open(tm["T_gt"]["path"]))["T_gt"], dtype=np.float64).reshape(4, 4)
        om = build_omega(yaml.safe_load(open("Registration/configs/gt_a/%s.yaml" % s)), seed=0)
        for t in tm["trials"]:
            if t["trial"] < 0:
                continue
            rows.append({"dataset": "D4", "stratum": s, "method": t["method"], "series": None, "trial": t["trial"],
                         "per": {"GA": per_trial(t["T_est"], t["P"], G, om, None)}})
    # D5
    # Ω は元の採点と同じ：R39 は各 config の build_omega（r39_bim_score.py）、R40 は err_zero の build_omega（r40_score.py）
    runs = [("R39_E2_centroid", "output/Registration/r39_bim/E2_centroid", False, "r39_bim/bim_E2_centroid"),
            ("R39_E2_planA", "output/Registration/r39_bim/E2_plan_correlate", False, "r39_bim/bim_E2_plan_correlate"),
            ("R39_E3_centroid", "output/Registration/r39_bim/E3_centroid", False, "r39_bim/bim_E3_centroid"),
            ("R39_E3_planA", "output/Registration/r39_bim/E3_plan_correlate", False, "r39_bim/bim_E3_plan_correlate"),
            ("R40_E2p_centroid", "output/Registration/r40_bim/E2p_centroid", True, "r40_bim/err_zero"),
            ("R40_E2p_planA", "output/Registration/r40_bim/E2p_plan_correlate", True, "r40_bim/err_zero")]
    for name in ["zero", "N_1cm", "N_2cm", "N_5cm", "L_5", "L_10", "L_20", "Obg_10", "Obg_25", "Obg_50",
                 "Owall_10", "Owall_25", "Owall_50", "S_0p5", "S_1p0", "S_2p0"]:
        runs.append(("R40_" + name, "output/Registration/r40_bim/err_" + name, False, "r40_bim/err_zero"))
    oms = {}
    for st, base, confounded, cfgname in runs:
        if cfgname not in oms:
            oms[cfgname] = build_omega(yaml.safe_load(open("Registration/configs/%s.yaml" % cfgname)))
        om = oms[cfgname]
        tm = json.load(open(base + "/trial_matrices.json"))
        G = np.asarray(json.load(open(tm["T_gt"]["path"]))["T_gt"], dtype=np.float64).reshape(4, 4)
        for t in tm["trials"]:
            if t["trial"] < 0:
                continue
            rows.append({"dataset": "D5", "stratum": st, "method": t["method"], "series": None, "trial": t["trial"],
                         "confounded": confounded, "per": {"GA": per_trial(t["T_est"], t["P"], G, om, None)}})
    return rows


# --------------------------------------------------------------------------- #
def main():
    os.makedirs(OUT, exist_ok=True)
    real = load_real()
    syn = load_synth()
    res = {"provenance": provenance(), "note": "結果確認後に設計した記述的な再解析（R41 §1）",
           "grid": {"a": A_GRID, "b": B_GRID.tolist(), "c": C_FIX}, "old": OLD}

    # ---- 旧基準の件数（G1 / 解析的正解）
    old = defaultdict(dict)
    for r in real:
        key = (r["dataset"], r["method"], r["series"] or "")
        old[key].setdefault("k", 0); old[key].setdefault("n", 0)
        old[key]["k"] += int(r["per"][1]["v"] and r["per"][1]["eR"] < 5 and r["per"][1]["eO"] < 0.1 and r["per"][1]["q"] < 0.05)
        old[key]["n"] += 1
    for r in syn:
        key = (r["dataset"], r["method"] + (" [" + r["stratum"] + "]" if r["dataset"] == "D5" else ""), "")
        p = r["per"]["GA"]
        old[key].setdefault("k", 0); old[key].setdefault("n", 0)
        old[key]["k"] += int(p["v"] and p["eR"] < 5 and p["eO"] < 0.1 and p["q"] < 0.05); old[key]["n"] += 1
    res["old_counts"] = {"|".join(k): v for k, v in old.items()}

    # ---- 曲線・分布（基準別）
    def groups(rows, crit_ids):
        g = defaultdict(list)
        for r in rows:
            g[(r["dataset"], r["method"], r["series"] or "", r.get("stratum") if r["dataset"] in ("D4", "D5") else "")].append(r)
        return g

    res["real"] = {}
    for (ds, m, ser, _), rs in groups(real, None).items():
        key = "%s|%s|%s" % (ds, m, ser)
        ent = {"N": len(rs), "per_criterion": {}}
        for j in range(1, 9):
            tr = [r["per"][j] for r in rs]
            ent["per_criterion"][j] = {"curve_eO": curve(tr, "eO"), "curve_eOB": curve(tr, "eOB"),
                                       "dist": {k: dist(tr, k) for k in ("eR", "tilt", "yaw", "eO", "eOB", "q")},
                                       "old_count": old_count(tr), "rep": {"%g|%g" % (a, b): F(tr, a, b) for a in REP_A for b in REP_B},
                                       "rep_BIM": ({"%g|%g" % (a, b): F(tr, a, b, key="eOB") for a in REP_A for b in REP_B}
                                                   if all(x["eOB"] is not None for x in tr) else None),
                                       "orientation": {o: sum(1 for x in tr if x["o"] == o) for o in ("ok", "90", "180", "invalid")}}
        # 基準感度（代表点）：G1、G1〜G6 の範囲、G1〜G8 の範囲
        sens = {}
        for a in REP_A:
            for b in REP_B:
                v = [ent["per_criterion"][j]["rep"]["%g|%g" % (a, b)] for j in range(1, 9)]
                sens["%g|%g" % (a, b)] = {"G1": v[0], "G1-6": [min(v[:6]), max(v[:6])], "G1-8": [min(v), max(v)]}
        ent["sensitivity"] = sens
        # 全点（固定条件の集合）
        if ds in ("D1", "D2"):
            ent["all_points"] = [{"stratum": r["stratum"], **{k: r["per"][1][k] for k in ("eR", "tilt", "yaw", "eO", "eOB", "q", "o")}} for r in rs]
        # 区間（D3 のみ：条件×系列×側）
        if ds == "D3":
            byc = defaultdict(list)
            for r in rs:
                byc[r["stratum"]].append(r["per"][1])
            ent["wilson_by_condition"] = {c: {"%g|%g" % (a, b): wil(v, a, b) for a in REP_A for b in REP_B} for c, v in byc.items()}
        res["real"][key] = ent

    # ---- 対応比較
    def paired(rows_a, rows_b, idkey):
        out = {"n_pairs": 0, "consistent_eO_1_6": 0, "consistent_eO_1_8": 0, "consistent_eR_1_6": 0, "consistent_eR_1_8": 0, "pairs": []}
        ib = {idkey(r): r for r in rows_b}
        for r in rows_a:
            s = ib.get(idkey(r))
            if s is None:
                continue
            dO = {j: r["per"][j]["eO"] - s["per"][j]["eO"] for j in range(1, 9)}
            dR = {j: r["per"][j]["eR"] - s["per"][j]["eR"] for j in range(1, 9)}
            same = lambda d, js: len({np.sign(d[j]) for j in js}) == 1
            out["n_pairs"] += 1
            out["consistent_eO_1_6"] += same(dO, range(1, 7)); out["consistent_eO_1_8"] += same(dO, range(1, 9))
            out["consistent_eR_1_6"] += same(dR, range(1, 7)); out["consistent_eR_1_8"] += same(dR, range(1, 9))
            out["pairs"].append({"id": list(idkey(r)), "d_eO": dO, "d_eR": dR})
        return out
    d2 = [r for r in real if r["dataset"] == "D2"]
    res["paired_D2"] = paired([r for r in d2 if r["method"] == "planA"], [r for r in d2 if r["method"] == "default"], lambda r: (r["stratum"],))
    d3 = [r for r in real if r["dataset"] == "D3"]
    res["paired_D3"] = {s: paired([r for r in d3 if r["method"] == "on" and r["series"] == s],
                                  [r for r in d3 if r["method"] == "off" and r["series"] == s],
                                  lambda r: (r["stratum"], r["trial"])) for s in ("S", "P")}

    # ---- 合成（D4・D5）
    res["synth"] = {}
    g = defaultdict(list)
    for r in syn:
        g[(r["dataset"], r["method"], r["stratum"])].append(r)
    for (ds, m, st), rs in g.items():
        tr = [r["per"]["GA"] for r in rs]
        res["synth"]["%s|%s|%s" % (ds, m, st)] = {
            "N": len(tr), "confounded": bool(rs[0].get("confounded", False)), "curve_eO": curve(tr, "eO"),
            "dist": {k: dist(tr, k) for k in ("eR", "tilt", "yaw", "eO", "q")}, "old_count": old_count(tr),
            "rep": {"%g|%g" % (a, b): F(tr, a, b) for a in REP_A for b in REP_B},
            "wilson": {"%g|%g" % (a, b): wil(tr, a, b) for a in REP_A for b in REP_B},
            "orientation": {o: sum(1 for x in tr if x["o"] == o) for o in ("ok", "90", "180", "invalid")}}
    # D4 のシーン等重みの全体
    res["D4_pooled_equal_scene_weight"] = {}
    for m in sorted({r["method"] for r in syn if r["dataset"] == "D4"}):
        cs = [res["synth"][k]["curve_eO"] for k in res["synth"] if k.startswith("D4|%s|" % m)]
        res["D4_pooled_equal_scene_weight"][m] = {"%g" % a: list(np.mean([c["%g" % a] for c in cs], axis=0)) for a in A_GRID}

    json.dump(res, open(os.path.join(OUT, "r41_rescore.json"), "w"), ensure_ascii=False, default=float)
    print("wrote", os.path.join(OUT, "r41_rescore.json"))


if __name__ == "__main__":
    main()
