"""R38 §3 — GT-A の全手法を、保存済みの $\\hat T$・$P$ から採点し直す。**手法は再実行しない。**

採点の形（回転 5°・縮尺 0.05・縮退は失敗、は共通。違うのは並進の量だけ）
    A. R27 主表の形：変換の並進ベクトルの差 ||t(T) - t(G P^-1)||（`sim3_errors` の `trans`）< 0.1 m
    B. R30・R36 の形：d_Ω(T, G P^-1, Ω)       … Ω（摂動前の座標）に、摂動後の source 用の変換を当てている
    C. 正しい形       ：d_Ω(T P, G, Ω)         … 摂動を戻してから比べる

B と C の関係：D = T P - G（摂動を戻した誤差の写像）とすると、
    C = RMS_{ω∈Ω} ||D ω||、  B = RMS_{ω∈Ω} ||D (P^-1 ω)||
B は、Ω を P^-1 で動かした点で同じ誤差を測っている。縮尺だけでなく、位置と向きも変わる。

対象：R27 の GT-A 実行（8 手法）、R36 の gt_a_release、R37 の gt_a_r37_off・gt_a_r37_on（`proposed` のみ）。
"""
from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np
import yaml

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "Registration"))
sys.path.insert(0, os.path.join(REPO, "Registration", "scripts"))
if not os.path.isdir("Registration"):
    os.chdir(REPO)
from criterion_verdict import build_omega, d_omega                   # noqa: E402
from failure_decomposition import provenance                          # noqa: E402
from r37_summary import wilson                                        # noqa: E402
from regbim import metrics                                            # noqa: E402

SCENES = ["room_0", "room_1", "room_2", "office_0", "office_1", "office_2", "office_3", "office_4"]
REG = {"rot_deg": 5.0, "trans": 0.1, "scale_ratio": 0.05}
RUNS = {"r27": "output/Registration/gt_a", "r36_release": "output/Registration/gt_a_release",
        "r37_off": "output/Registration/gt_a_r37_off", "r37_on": "output/Registration/gt_a_r37_on"}


def q(a):
    a = np.asarray([x for x in a if np.isfinite(x)], dtype=float)
    if not len(a):
        return None
    return {"median": float(np.median(a)), "p90": float(np.percentile(a, 90)),
            "max": float(a.max()), "min": float(a.min()), "n_finite": int(len(a))}


def score_run(base, scene, omega):
    p = "%s/%s/trial_matrices.json" % (base, scene)
    if not os.path.exists(p):
        return None
    tm = json.load(open(p))
    G = np.asarray(json.load(open(tm["T_gt"]["path"]))["T_gt"], dtype=np.float64).reshape(4, 4)
    # R27 が記録した判定（trials.csv の gt_success）と照合する
    rec = {}
    tc = "%s/%s/trials.csv" % (base, scene)
    if os.path.exists(tc):
        for r in csv.DictReader(open(tc)):
            rec[(r["method"], int(r["trial"]))] = r
    cen = omega.mean(axis=0)
    out = []
    for t in tm["trials"]:
        if t["trial"] < 0:
            continue
        T = np.asarray(t["T_est"], dtype=np.float64)
        P = np.asarray(t["P"], dtype=np.float64)
        E = G @ metrics.invert_sim3(P)
        e = metrics.sim3_errors(T, E)
        nf = bool(e.get("non_finite"))
        dB = float("inf") if nf else d_omega(T, E, omega)
        dC = float("inf") if nf else d_omega(T @ P, G, omega)
        base_ok = (not e["degenerate"]) and e["rot_deg"] < REG["rot_deg"] and e["scale_ratio"] < REG["scale_ratio"]
        Rp, tp, sp = metrics.decompose_sim3(P)
        Pi = metrics.invert_sim3(P)
        r0 = rec.get((t["method"], t["trial"]))
        out.append({
            "scene": scene, "method": t["method"], "trial": t["trial"],
            "rot_deg": e["rot_deg"], "scale_ratio": e["scale_ratio"], "degenerate": bool(e["degenerate"]),
            "trans_raw": e["trans"], "d_old": dB, "d_new": dC,
            "ok_A": bool(base_ok and e["trans"] < REG["trans"]),
            "ok_B": bool(base_ok and dB < REG["trans"]),
            "ok_C": bool(base_ok and dC < REG["trans"]),
            "recorded_gt_success": None if r0 is None else (r0["gt_success"] == "True"),
            "P_rot_deg": float(np.degrees(np.arccos(np.clip((np.trace(Rp) - 1) / 2, -1, 1)))),
            "P_scale": float(sp), "P_trans_m": float(np.linalg.norm(tp)),
            # Ω の重心が P^-1 でどれだけ動くか（B が別の場所で測っている量）
            "omega_centroid_shift_m": float(np.linalg.norm(metrics.apply_sim3(Pi, cen[None])[0] - cen)),
        })
    return out


def main():
    res = {"provenance": provenance(), "runs": {}}
    omegas = {}
    for s in SCENES:
        omegas[s] = build_omega(yaml.safe_load(open("Registration/configs/gt_a/%s.yaml" % s)), seed=0)
    for run, base in RUNS.items():
        rows = []
        for s in SCENES:
            r = score_run(base, s, omegas[s])
            if r is None:
                print("%s %s: trial_matrices が無い" % (run, s))
                continue
            rows += r
        res["runs"][run] = rows
        print("\n## %s（%s）" % (run, base))
        print("%-22s %5s | %-26s | %-26s | %-26s | 変化(B→C) | R27記録との一致(A)"
              % ("手法", "n", "A 並進ベクトル", "B 旧 d_Ω", "C 正しい d_Ω"))
        for m in sorted({x["method"] for x in rows}):
            v = [x for x in rows if x["method"] == m]
            n = len(v)
            k = {f: sum(x["ok_" + f] for x in v) for f in "ABC"}
            ci = {f: wilson(k[f], n) for f in "ABC"}
            p2f = sum(1 for x in v if x["ok_B"] and not x["ok_C"])
            f2p = sum(1 for x in v if x["ok_C"] and not x["ok_B"])
            a2c_p2f = sum(1 for x in v if x["ok_A"] and not x["ok_C"])
            a2c_f2p = sum(1 for x in v if x["ok_C"] and not x["ok_A"])
            recs = [x for x in v if x["recorded_gt_success"] is not None]
            agree = sum(1 for x in recs if x["recorded_gt_success"] == x["ok_A"])
            print("%-22s %5d | %3d (%.1f%%) [%.1f,%.1f] | %3d (%.1f%%) [%.1f,%.1f] | %3d (%.1f%%) [%.1f,%.1f] | B→C 合→否 %d 否→合 %d / A→C 合→否 %d 否→合 %d | %s"
                  % (m, n, k["A"], 100 * k["A"] / n, 100 * ci["A"][0], 100 * ci["A"][1],
                     k["B"], 100 * k["B"] / n, 100 * ci["B"][0], 100 * ci["B"][1],
                     k["C"], 100 * k["C"] / n, 100 * ci["C"][0], 100 * ci["C"][1],
                     p2f, f2p, a2c_p2f, a2c_f2p,
                     "%d/%d" % (agree, len(recs)) if recs else "—"))
            res.setdefault("summary", {}).setdefault(run, {})[m] = {
                "n": n, "k": k, "ci": ci,
                "B_to_C": {"pass_to_fail": p2f, "fail_to_pass": f2p},
                "A_to_C": {"pass_to_fail": a2c_p2f, "fail_to_pass": a2c_f2p},
                "recorded_agree_A": [agree, len(recs)],
                "trans_raw": q([x["trans_raw"] for x in v]), "d_old": q([x["d_old"] for x in v]),
                "d_new": q([x["d_new"] for x in v]),
                "n_degenerate": sum(x["degenerate"] for x in v),
                "rot_ok": sum(1 for x in v if x["rot_deg"] < 5 and not x["degenerate"]),
                "scale_ok": sum(1 for x in v if x["scale_ratio"] < 0.05 and not x["degenerate"])}
    # 比 B/C と摂動の関係（R27 の proposed）
    v = [x for x in res["runs"]["r27"] if x["method"] == "proposed" and x["d_new"] > 0]
    ratio = np.array([x["d_old"] / x["d_new"] for x in v])
    from scipy.stats import spearmanr
    rel = {k: float(spearmanr([x[k] if k != "P_scale" else abs(np.log(x[k])) for x in v], ratio)[0])
           for k in ("P_scale", "P_rot_deg", "P_trans_m", "omega_centroid_shift_m")}
    res["ratio_proposed"] = {"q": q(ratio), "spearman": rel,
                             "scale_range_bound": [float(np.exp(-0.4)), float(np.exp(0.4))],
                             "n_outside_scale_bound": int(np.sum((ratio > np.exp(0.4)) | (ratio < np.exp(-0.4))))}
    i = int(np.argmax(ratio))
    res["ratio_example"] = dict(v[i], ratio=float(ratio[i]))
    print("\n## 比 B/C（R27 proposed、%d 試行）" % len(v), res["ratio_proposed"])
    print("最大の例:", {k: res["ratio_example"][k] for k in ("scene", "trial", "d_old", "d_new", "ratio", "P_scale", "P_rot_deg", "P_trans_m", "omega_centroid_shift_m")})
    json.dump(res, open("Registration/output/diag/r38_gta_rescore.json", "w"), ensure_ascii=False, default=float)
    print("wrote Registration/output/diag/r38_gta_rescore.json")


if __name__ == "__main__":
    main()
