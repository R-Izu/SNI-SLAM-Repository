"""R51 段階0 §3-2・§7：候補の曖昧さの表（sni-slam の環境）。既存の出力から計算する。**位置合わせは回さない。**

対象：R49 の完全（`~/rizu/r49_data/results/<data>/full/`：synth・real・bim）、R50 の採点だけクラスなし・参照のクラスを幾何から作る
（`~/rizu/r50_data/results/<data>/{score_only,geo}/`：synth・real）
各試行の記録 `diag`（`r49_run.py:121-122`・`r50_run.py:59-60`）：after_icp（通常の精緻化の 4 候補の変換）、scores（その ρ、
`proposed.py:421` で小数 5 桁に丸めた値）、short（短い精緻化の 16 候補の変換）

定義（指示書 §3-2・§7 のとおり。ここで足した細部は ＊）
- 候補 T の採点：`per_trial(T, P, G, Ω)`。正しい＝有効かつ e_R < 45° かつ e_Ω < 1.0 m（大きな失敗の線の内側）。実データは G1（G2〜G8 も）
- ρ1〜ρ4：4 候補の ρ を大きい順。Δρ = ρ1 − ρ2。順位は ρ の大きい順（同じ値は記録の順、＊）。1 位の候補 = 手法が選んだ候補
- 正しい候補の順位：線の内側の候補のうち最上位の順位（無ければ なし）
- 1 位と正しい候補の違い：回転の角（両者の回転の相対の角）、ヨーの差（`tilt_yaw` の yaw を (−180°, 180°] に折り返す）、
  並進の差（＊ Ω の重心 c を (T P) で参照の座標に置いたときの位置の差 |(T1 P)c − (Tc P)c|）
- 同値：4 候補を節点とし、回転の角 < 5° かつ並進の差（上と同じ ＊）< 0.3 m の対を辺で結んだ連結成分を 1 組とする。
  組の数と、各組の中の最大の回転の角・並進の差・縮尺の比の差 |s_i/s_j − 1|
- AUC：大きな失敗を陽性、Δρ が小さいほど陽性と予測（Mann–Whitney、同じ値は 0.5）。CI：試行のブートストラップ 2,000 回（seed 0）と、
  シーン（スキャン）単位のブートストラップ 2,000 回（seed 0）。陽性か陰性が無い標本は数えて飛ばす
- 正しい候補を含む割合：Wilson 95% CI と、シーン単位のブートストラップ（2,000 回、seed 0）
- ヨーの差の区分：0°・+90°・−90°・180° の ±10° と、それ以外
- 16 候補の出所：16 候補にも 4 候補にも無い／16 候補にはあるが 4 候補に無い／4 候補にある
出力：試行ごと `~/rizu/r51_data/stage0_trials.json`、まとめ `Registration/output/diag/r51/r51_stage0.json`
"""
import json, os, sys
from collections import Counter
import numpy as np

REPO = os.path.expanduser("~/rizu/SNI-SLAM")
for p in ("Registration", "Registration/scripts", "Registration/scripts/r49"):
    sys.path.insert(0, os.path.join(REPO, p))
os.chdir(REPO)
from regbim import metrics                                      # noqa: E402
from failure_decomposition import provenance                    # noqa: E402
from r41_rescore import per_trial                               # noqa: E402
from r36_rotation_diag import tilt_yaw                          # noqa: E402
import r49_run as RR                                            # noqa: E402
from r49_score import truth, big                                # noqa: E402

SETS = [("full", os.path.expanduser("~/rizu/r49_data/results"), ("synth", "real", "bim")),
        ("score_only", os.path.expanduser("~/rizu/r50_data/results"), ("synth", "real")),
        ("geo", os.path.expanduser("~/rizu/r50_data/results"), ("synth", "real"))]
NB = 2000


def ok(p):
    return bool(p["v"] and p["eR"] < 45.0 and p["eO"] < 1.0)


def rel(Ta, Tb, P, c):
    A, B = np.asarray(Ta) @ P, np.asarray(Tb) @ P
    Ra, _, sa = metrics.decompose_sim3(A); Rb, _, sb = metrics.decompose_sim3(B)
    ang = metrics.rotation_error_deg(Ra, Rb)
    yaw = float(tilt_yaw(Ra, Rb)["yaw_deg"]); yaw = yaw - 360.0 if yaw > 180.0 else (yaw + 360.0 if yaw <= -180.0 else yaw)
    tr = float(np.linalg.norm(metrics.apply_sim3(A, c[None])[0] - metrics.apply_sim3(B, c[None])[0]))
    return ang, yaw, tr, float(abs(sa / sb - 1.0))


def groups(Ts, P, c):
    n = len(Ts); par = list(range(n))
    def f(i):
        while par[i] != i:
            i = par[i]
        return i
    for i in range(n):
        for j in range(i + 1, n):
            a, _, t, _ = rel(Ts[i], Ts[j], P, c)
            if a < 5.0 and t < 0.3:
                par[f(i)] = f(j)
    comp = {}
    for i in range(n):
        comp.setdefault(f(i), []).append(i)
    out = []
    for g in comp.values():
        m = [rel(Ts[i], Ts[j], P, c) for i in g for j in g if i < j]
        out.append({"members": g, "max_rot": max((x[0] for x in m), default=0.0), "max_trans": max((x[2] for x in m), default=0.0),
                    "max_scale": max((x[3] for x in m), default=0.0)})
    return out


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return None
    # 陽性ほど Δρ が小さいと予測：score = −Δρ
    s_p, s_n = -pos, -neg
    gt = (s_p[:, None] > s_n[None, :]).sum(); eq = (s_p[:, None] == s_n[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(pos) * len(neg)))


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return [None, None]
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(c - h), float(c + h)]


def boot_trials(rows, stat, rng):
    vals, skipped = [], 0
    for _ in range(NB):
        s = [rows[i] for i in rng.integers(0, len(rows), len(rows))]
        v = stat(s)
        if v is None:
            skipped += 1
        else:
            vals.append(v)
    return {"ci95": [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if vals else None, "n_skipped": skipped}


def boot_scenes(rows, stat, rng):
    sc = sorted({r["target_scene"] for r in rows}); by = {s: [r for r in rows if r["target_scene"] == s] for s in sc}
    vals, skipped = [], 0
    for _ in range(NB):
        s = [r for k in rng.integers(0, len(sc), len(sc)) for r in by[sc[k]]]
        v = stat(s)
        if v is None:
            skipped += 1
        else:
            vals.append(v)
    return {"ci95": [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if vals else None, "n_skipped": skipped,
            "per_scene": {s: stat(by[s]) for s in sc}}


def q(v):
    v = [x for x in v if x is not None]
    return None if not v else {"n": len(v), "p5": float(np.percentile(v, 5)), "median": float(np.median(v)), "p95": float(np.percentile(v, 95))}


def stat_auc(rs):
    return auc([r["drho"] for r in rs if r["fail"]], [r["drho"] for r in rs if not r["fail"]])


def stat_contains(rs):
    return float(np.mean([r["contains"]["1"] for r in rs])) if rs else None


def yaw_bin(y):
    for name, c in (("0", 0.0), ("+90", 90.0), ("-90", -90.0)):
        if abs(y - c) <= 10.0:
            return name
    return "180" if abs(y) >= 170.0 else "other"


def main():
    out_trials = []; summ = {"provenance": provenance()}
    for m, base, datas in SETS:
        for data in datas:
            rows = []
            for target in RR.targets(data):
                Gs, om, _ = truth(data, target); c = om.mean(0)
                res = json.load(open(os.path.join(base, data, m, target + ".json")))["rows"]
                for k in range(RR.n_trials(data)):
                    r = res[str(k)]; d = r["diag"]; P = np.asarray(r["P"], dtype=np.float64)
                    sc = np.asarray(d["scores"], dtype=float); order = list(np.argsort(-sc, kind="stable"))
                    Ts = d["after_icp"]
                    corr = {j: [ok(per_trial(T, P, G, om, None)) for T in Ts] for j, G in Gs.items()}
                    corr16 = [ok(per_trial(T, P, Gs["1"], om, None)) for T in d["short"]]
                    fail = big(per_trial(r["T_est"], P, Gs["1"], om, None))
                    ranks = [i + 1 for i, ci in enumerate(order) if corr["1"][ci]]
                    row = {"method": m, "data": data, "target": target, "target_scene": target.split("__")[0] if data == "real" else target,
                           "trial": k, "fail": bool(fail), "rho_sorted": [float(sc[i]) for i in order],
                           "drho": float(sc[order[0]] - sc[order[1]]) if len(order) > 1 else None,
                           "cand_correct_G1_in_score_order": [corr["1"][i] for i in order],
                           "contains": {j: bool(any(v)) for j, v in corr.items()}, "rank_correct": ranks[0] if ranks else None,
                           "short16_contains": bool(any(corr16)), "groups": groups(Ts, P, c)}
                    if ranks:
                        a, y, t, s = rel(Ts[order[0]], Ts[order[ranks[0] - 1]], P, c)
                        row["top_vs_correct"] = {"rot_deg": a, "yaw_deg": y, "trans_m": t, "scale_rel": s}
                    rows.append(row)
            out_trials += rows
            rng = np.random.default_rng(0)
            n = len(rows); nf = sum(r["fail"] for r in rows); kc = sum(r["contains"]["1"] for r in rows)
            fails = [r for r in rows if r["fail"]]
            ent = {"n": n, "n_fail": nf, "contains_G1": {"k": kc, "frac": kc / n, "wilson95": wilson(kc, n),
                                                       "scene_boot": boot_scenes(rows, stat_contains, np.random.default_rng(0))},
                   "rank_correct": dict(Counter(str(r["rank_correct"]) for r in rows)),
                   "auc": {"value": stat_auc(rows), "trial_boot": boot_trials(rows, stat_auc, rng),
                           "scene_boot": boot_scenes(rows, stat_auc, np.random.default_rng(0))},
                   "drho_fail": q([r["drho"] for r in fails]), "drho_success": q([r["drho"] for r in rows if not r["fail"]]),
                   "drho_zero": {"fail": sum(r["drho"] == 0 for r in fails), "success": sum(r["drho"] == 0 for r in rows if not r["fail"])},
                   "fail_top_vs_correct": {"n_with_correct": sum("top_vs_correct" in r for r in fails),
                                           "yaw_bins": dict(Counter(yaw_bin(r["top_vs_correct"]["yaw_deg"]) for r in fails if "top_vs_correct" in r)),
                                           "rot_deg": q([r["top_vs_correct"]["rot_deg"] for r in fails if "top_vs_correct" in r]),
                                           "trans_m": q([r["top_vs_correct"]["trans_m"] for r in fails if "top_vs_correct" in r])},
                   "origin16": {"none": sum(not r["short16_contains"] and not r["contains"]["1"] for r in rows),
                                "in16_not4": sum(r["short16_contains"] and not r["contains"]["1"] for r in rows),
                                "in4": sum(r["contains"]["1"] for r in rows),
                                "fail_none": sum(not r["short16_contains"] and not r["contains"]["1"] for r in fails),
                                "fail_in16_not4": sum(r["short16_contains"] and not r["contains"]["1"] for r in fails),
                                "fail_in4": sum(r["contains"]["1"] for r in fails)},
                   "groups_count": dict(Counter(str(len(r["groups"])) for r in rows)),
                   "groups_count_fail": dict(Counter(str(len(r["groups"])) for r in fails)),
                   "group_max_rot": q([g["max_rot"] for r in rows for g in r["groups"] if len(g["members"]) > 1]),
                   "group_max_trans": q([g["max_trans"] for r in rows for g in r["groups"] if len(g["members"]) > 1]),
                   "group_max_scale": q([g["max_scale"] for r in rows for g in r["groups"] if len(g["members"]) > 1])}
            if data == "real":
                ent["contains_G2_G8"] = {j: {"k": sum(r["contains"][j] for r in rows), "frac": sum(r["contains"][j] for r in rows) / n}
                                         for j in sorted(rows[0]["contains"]) if j != "1"}
            summ["%s|%s" % (data, m)] = ent
            print(data, m, "n", n, "fail", nf, "contains %.3f" % (kc / n), "auc", ent["auc"]["value"], ent["auc"]["trial_boot"]["ci95"], flush=True)
    os.makedirs(os.path.expanduser("~/rizu/r51_data"), exist_ok=True)
    json.dump(out_trials, open(os.path.expanduser("~/rizu/r51_data/stage0_trials.json"), "w"), default=float)
    os.makedirs("Registration/output/diag/r51", exist_ok=True)
    json.dump(summ, open("Registration/output/diag/r51/r51_stage0.json", "w"), ensure_ascii=False, indent=1, default=float)


if __name__ == "__main__":
    main()
