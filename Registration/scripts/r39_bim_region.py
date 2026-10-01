"""R39 §3 — BIM が覆う範囲での副指標 d_{Ω_BIM}（採点し直しのみ。手法は再実行しない）。

**この副指標は、結果を見た後で main が足したものである**（R39 §3-4）。主基準は変えない。

定義（R39 §3-1）
- Ω_BIM：Ω（`omega.npz`、seed 0 の source の構造点 2 万点）のうち、**G1 で置いたとき**に、
  その条件の BIM 参照の点から**水平距離 0.3 m 以内**にある点（R38 §4-1 の「部屋」と同じ規則）
- d_{Ω_BIM}(T, G)：Ω_BIM で測った d_Ω。形は正しい形 C（摂動は戻してから比べる）
- Ω_BIM は条件ごとに 1 回作って保存し、以後固定（`r39_omega_bim.npz`）。どの基準で採点するときも同じ点集合

対象：R35 の 16 出力（既定・案A）、R37 の 160 対（off・on）。6 基準すべて。
閾値は主基準と同じ（回転 5°・0.1 m・縮尺 0.05、縮退は失敗）。**別の閾値を探さない。**
"""
from __future__ import annotations

import copy
import json
import os
import sys
from collections import defaultdict

import numpy as np
import yaml
from scipy.spatial import cKDTree

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "Registration"))
sys.path.insert(0, os.path.join(REPO, "Registration", "scripts"))
if not os.path.isdir("Registration"):
    os.chdir(REPO)
from criterion_verdict import d_omega                        # noqa: E402
from failure_decomposition import provenance                 # noqa: E402
from regbim import io_utils, metrics                         # noqa: E402

TARGETS = ["m3_cor_a__E2", "m3_cor_a__E3", "m3_cor_b__E2", "m3_cor_b__E3",
           "m3_cor_c__E2", "m3_cor_c__E3", "m3_cor_d__E2", "m3_cor_d__E3"]
ROOM_DIST = 0.3
TH = {"rot_deg": 5.0, "trans": 0.1, "scale_ratio": 0.05}
OMEGA_BIM_NPZ = "Registration/output/diag/r39_omega_bim.npz"


def crits(scene):
    sp = json.load(open("Registration/output/diag/criterion_spread.json"))["scenes"][scene]
    return [(f["id"], np.asarray(f["T"], dtype=np.float64)) for f in sp["results"] if "T" in f]


def long_sd(p):
    c = np.cov((p - p.mean(0)).T)
    return float(np.sqrt(np.linalg.eigvalsh(c)[-1]))


def score(T, cs, om, omb):
    R, _, s = metrics.decompose_sim3(T)
    degen = bool(metrics.is_degenerate_sim3(T))
    per = []
    for j, G in cs:
        RG, _, sG = metrics.decompose_sim3(G)
        rot = metrics.rotation_error_deg(R, RG); sc = float(abs(s / sG - 1))
        d, db = d_omega(T, G, om), d_omega(T, G, omb)
        base = (not degen) and rot < TH["rot_deg"] and sc < TH["scale_ratio"]
        per.append({"id": j, "rot_deg": rot, "scale_ratio": sc, "d": d, "d_bim": db,
                    "ok": bool(base and d < TH["trans"]), "ok_bim": bool(base and db < TH["trans"])})
    def three(key):
        n = sum(p[key] for p in per)
        return "all_pass" if n == len(per) else ("all_fail" if n == 0 else "criterion_dependent")
    p1 = next(p for p in per if p["id"] == 1)
    return {"per": per, "primary": p1, "three": three("ok"), "three_bim": three("ok_bim")}


def main():
    om_all = np.load("Registration/output/diag/omega.npz")
    omb, info = {}, {}
    for t in TARGETS:
        scene = t.split("__")[0]
        cfg = copy.deepcopy(yaml.safe_load(open("Registration/configs/realdata/%s.yaml" % t)))
        dst = io_utils.load_reference_cloud(cfg)
        G1 = dict(crits(scene))[1]
        om = om_all[scene]
        P = metrics.apply_sim3(G1, om)
        m = cKDTree(dst.points[:, :2]).query(P[:, :2], k=1, workers=-1)[0] <= ROOM_DIST
        omb[t] = om[m]
        info[t] = {"n_omega": int(len(om)), "n_omega_bim": int(m.sum()), "frac": float(m.mean()),
                   "long_sd_omega_m": long_sd(om), "long_sd_omega_bim_m": long_sd(om[m])}
        print("%-14s Ω_BIM %5d / %d 点（%.1f%%） 長軸 SD：Ω %.2f m → Ω_BIM %.2f m"
              % (t, m.sum(), len(om), 100 * m.mean(), info[t]["long_sd_omega_m"], info[t]["long_sd_omega_bim_m"]))
    np.savez_compressed(OMEGA_BIM_NPZ, **omb)

    res = {"provenance": provenance(), "note": "d_Ω_BIM は結果を見た後で main が足した副指標（R39 §3-4）。主基準は変えない",
           "omega_bim": info, "r35": [], "r37": []}
    for r in json.load(open("Registration/output/diag/r35_runs.json"))["rows"]:
        t = "%s__%s" % (r["scene"], r["cond"])
        s = score(np.asarray(r["T_est"]), crits(r["scene"]), om_all[r["scene"]], omb[t])
        res["r35"].append({"target": t, "mode": r["mode"], **s})
    for r in json.load(open("Registration/output/diag/r37_replicate.json"))["rows"]:
        t = r["target"]; scene = t.split("__")[0]; P = np.asarray(r["P"]) if r["P"] else np.eye(4)
        for side in ("off", "on"):
            s = score(np.asarray(r["T_" + side]) @ P, crits(scene), om_all[scene], omb[t])
            res["r37"].append({"target": t, "series": r["series"], "k": r["k"], "side": side, **s})
    json.dump(res, open("Registration/output/diag/r39_bim_region.json", "w"), ensure_ascii=False, default=float)

    print("\n## R35（16 出力、主基準 G1）")
    for x in res["r35"]:
        p = x["primary"]
        print("%-14s %-14s dΩ %8.4f  dΩ_BIM %8.4f | 主基準 %s 副指標 %s | 3区分 %s / 副 %s"
              % (x["target"], x["mode"], p["d"], p["d_bim"], "○" if p["ok"] else "×",
                 "○" if p["ok_bim"] else "×", x["three"], x["three_bim"]))
    print("\n## R37（160 対、主基準 G1）")
    g = defaultdict(list)
    for x in res["r37"]:
        g[(x["target"], x["side"])].append(x)
    for (t, side), v in sorted(g.items()):
        d = [x["primary"]["d"] for x in v]; db = [x["primary"]["d_bim"] for x in v]
        c3 = defaultdict(int); cb = defaultdict(int)
        for x in v:
            c3[x["three"]] += 1; cb[x["three_bim"]] += 1
        print("%-14s %-3s n=%d dΩ 中央 %.3f [%.3f, %.3f] | dΩ_BIM 中央 %.3f [%.3f, %.3f] | 主基準 %d 副 %d | 3区分 %s / 副 %s"
              % (t, side, len(v), np.median(d), min(d), max(d), np.median(db), min(db), max(db),
                 sum(x["primary"]["ok"] for x in v), sum(x["primary"]["ok_bim"] for x in v), dict(c3), dict(cb)))


if __name__ == "__main__":
    main()
