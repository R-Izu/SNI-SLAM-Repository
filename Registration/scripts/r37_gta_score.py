"""R37 §5 — GT-A の同一 commit での off / on（`proposed`、centroid 経路）を採点する。

採点は2通り出す：
- **正しい形**：摂動を戻した $\\hat T P$ と $G$ を、Ω（seed 0 の source 座標）で比べる
- **旧い形**（R30・R36 の `threshold_curve.gt_a_rows` と同じ）：$\\hat T$ と $GP^{-1}$ を、摂動前の Ω で比べる
  （2 つの変換の差の量としては成り立つが、測る点の集合が摂動の縮尺ぶん歪む）

成功は事前登録値（回転 5°・d_Ω 0.1 m・縮尺 0.05、縮退は失敗）。回転・縮尺は2通りで同じ。
失敗 3 種のうち GT-A で見られるのは縮尺の潰れ・取り違えだけ（反復の記録が無い）。
"""
from __future__ import annotations

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


def rows(base, s, omega):
    tm = json.load(open("%s/%s/trial_matrices.json" % (base, s)))
    G = np.asarray(json.load(open(tm["T_gt"]["path"]))["T_gt"], dtype=np.float64).reshape(4, 4)
    out = {}
    for t in tm["trials"]:
        if t["method"] != "proposed" or t["trial"] < 0:
            continue
        T = np.asarray(t["T_est"], dtype=np.float64)
        P = np.asarray(t["P"], dtype=np.float64)
        E = G @ metrics.invert_sim3(P)
        e = metrics.sim3_errors(T, E)
        if e.get("non_finite"):
            d_new = d_old = float("inf")
        else:
            d_new = d_omega(T @ P, G, omega)
            d_old = d_omega(T, E, omega)
        _, _, s1 = metrics.decompose_sim3(T); _, _, sE = metrics.decompose_sim3(E)
        base_ok = (not e["degenerate"] and e["rot_deg"] < REG["rot_deg"]
                   and e["scale_ratio"] < REG["scale_ratio"])
        out[t["trial"]] = {"rot_deg": e["rot_deg"], "scale_ratio": e["scale_ratio"],
                           "d_new": d_new, "d_old": d_old, "degenerate": bool(e["degenerate"]),
                           "ok_new": bool(base_ok and d_new < REG["trans"]),
                           "ok_old": bool(base_ok and d_old < REG["trans"]),
                           "fail_scale_collapse": bool(e["degenerate"] or not (0.5 <= s1 / sE <= 2.0)),
                           "fail_flip": bool(e["rot_deg"] > 45.0), "P": P}
    return out


def q(a):
    a = np.asarray(a, dtype=float)
    return {"median": float(np.median(a)), "p90": float(np.percentile(a, 90)),
            "max": float(a.max()), "min": float(a.min())}


def main():
    res = {"provenance": provenance(), "scenes": {}}
    allv = {"off": [], "on": []}
    same_P = True
    for s in SCENES:
        cfg = yaml.safe_load(open("Registration/configs/gt_a/%s.yaml" % s))
        omega = build_omega(cfg, seed=0)
        off = rows("output/Registration/gt_a_r37_off", s, omega)
        on = rows("output/Registration/gt_a_r37_on", s, omega)
        r27 = rows("output/Registration/gt_a", s, omega)
        for k in on:
            same_P &= bool(np.array_equal(on[k]["P"], off[k]["P"])
                           and np.array_equal(on[k]["P"], r27[k]["P"]))
        for side, v in (("off", off), ("on", on)):
            allv[side] += [dict(x, scene=s, trial=k) for k, x in v.items()]
        res["scenes"][s] = {
            side: {"ok_new": sum(x["ok_new"] for x in v.values()),
                   "ok_old": sum(x["ok_old"] for x in v.values()), "n": len(v),
                   "d_new": q([x["d_new"] for x in v.values()]),
                   "rot": q([x["rot_deg"] for x in v.values()])}
            for side, v in (("off", off), ("on", on))}
        # R27 の旧実行と、今回の off の一致（同じ commit でなくても同じ解か）
        res["scenes"][s]["off_vs_r27_max_rot_diff_deg"] = max(
            abs(off[k]["rot_deg"] - r27[k]["rot_deg"]) for k in off)
        o, n = res["scenes"][s]["off"], res["scenes"][s]["on"]
        print("%-9s off %d/%d (新 d 最大 %.4f 回転中央 %.3f) | on %d/%d (新 d 最大 %.4f 回転中央 %.3f) | off と R27 の回転差 最大 %.2e"
              % (s, o["ok_new"], o["n"], o["d_new"]["max"], o["rot"]["median"],
                 n["ok_new"], n["n"], n["d_new"]["max"], n["rot"]["median"],
                 res["scenes"][s]["off_vs_r27_max_rot_diff_deg"]), flush=True)
    tot = {}
    for side in ("off", "on"):
        v = allv[side]
        k_new, k_old = sum(x["ok_new"] for x in v), sum(x["ok_old"] for x in v)
        tot[side] = {"ok_new": k_new, "ok_old": k_old, "n": len(v),
                     "ci_new": wilson(k_new, len(v)),
                     "d_new": q([x["d_new"] for x in v]), "d_old": q([x["d_old"] for x in v]),
                     "rot": q([x["rot_deg"] for x in v]),
                     "fail_scale_collapse": sum(x["fail_scale_collapse"] for x in v),
                     "fail_flip": sum(x["fail_flip"] for x in v),
                     "degenerate": sum(x["degenerate"] for x in v)}
    key = lambda x: (x["scene"], x["trial"])
    offd = {key(x): x for x in allv["off"]}
    dd = [x["d_new"] - offd[key(x)]["d_new"] for x in allv["on"]]
    dr = [x["rot_deg"] - offd[key(x)]["rot_deg"] for x in allv["on"]]
    tot["paired_on_minus_off"] = {"d_new": q(dd), "rot": q(dr),
                                  "n_d_worse": sum(1 for x in dd if x > 0),
                                  "n_rot_worse": sum(1 for x in dr if x > 0)}
    tot["same_P_all"] = same_P
    ratio = [x["d_old"] / x["d_new"] for x in allv["off"] if x["d_new"] > 0]
    tot["legacy_over_new_ratio"] = q(ratio)
    res["totals"] = tot
    print(json.dumps({k: v for k, v in tot.items()}, indent=1, ensure_ascii=False, default=float))
    json.dump(res, open("Registration/output/diag/r37_gta_score.json", "w"), indent=1,
              ensure_ascii=False, default=float)


if __name__ == "__main__":
    main()
