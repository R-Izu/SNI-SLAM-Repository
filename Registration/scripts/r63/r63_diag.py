"""R63 §7-2 の 1：上向きと縮尺の推定の確認（sni-slam の環境）。**回す前に commit する。位置合わせは回さない。**

(a) 上向き：R63 の 40 試行（完全・b-2・f 1p0）と同じ点（source は `prepare(transform_cloud(src, P), cfg)`、参照は `prepare(dst, cfg)`）に、
    本手法と同じ `rotation.estimate_gravity_axis(cloud, cfg)` を当て、正解の上向きとの角を出す。
    正解の上向き：参照は IFC の e_z。source は (A P) の回転 R で、R⁻¹ e_z（P で動かした source の座標での IFC の上向き）
    角は符号つき（推定の向きは天井が床より上になるように決められているので、180° 近くは上下が逆）
(b) 縮尺：40 試行の `diag.gen`（16 候補）の縮尺と、正解の縮尺 s_true = s_A / s_P（T は P で動かした source → IFC）の比。
    `plan_correlate` の縮尺の格子の範囲（config の `proposed.plan_correlate`、無ければ既定 0.3〜3.0・170 点）に s_true が入るか
(c) 法線：入力 b-2 の f 1p0（P なし）を `prepare` した点の、床・天井のラベルの点の法線と、正解の上向き（A⁻¹ の e_z）の角。
    折り返した角 min(θ, 180° − θ) の分布。参照 4 つの床・天井の点の法線と e_z の角も
出力：`~/rizu/r63_data/diag.json`
"""
import json, os, sys
import numpy as np

REPO = os.path.expanduser("~/rizu/SNI-SLAM")
for p in ("Registration", "Registration/scripts", "Registration/scripts/r49"):
    sys.path.insert(0, os.path.join(REPO, p))
os.chdir(REPO)
import yaml                                                    # noqa: E402
from regbim import io_utils, metrics, preprocess, rotation     # noqa: E402
from regbim.labels import NAME_TO_ID                           # noqa: E402

CO = os.path.expanduser("~/rizu/conslam/out")
REFS = ["S1_small", "S1_large", "S3_small", "S3_large"]
A = np.array(json.load(open(os.path.join(CO, "extra.json")))["b_normalized"]["A"])
EZ = np.array([0.0, 0.0, 1.0])


def ang(u, v):
    u = u / np.linalg.norm(u); v = v / np.linalg.norm(v)
    return float(np.degrees(np.arccos(np.clip(u @ v, -1, 1))))


def q(v):
    v = np.asarray(v, dtype=float)
    return {"n": int(len(v)), "median": float(np.median(v)), "p5": float(np.percentile(v, 5)), "p25": float(np.percentile(v, 25)),
            "p75": float(np.percentile(v, 75)), "p95": float(np.percentile(v, 95))}


def main():
    res = {"a": {}, "b": {}, "c": {}}
    from regbim.plan_correlate import DEFAULTS
    for ref in REFS:
        cfg = yaml.safe_load(open("Registration/configs/conslam/b2_f1p0__%s.yaml" % ref))
        src = io_utils.load_source_cloud(cfg); dst = io_utils.load_reference_cloud(cfg); dp = preprocess.prepare(dst, cfg)
        up_ref = rotation.estimate_gravity_axis(dp, cfg)
        rows = json.load(open(os.path.expanduser("~/rizu/r63_data/results/full/b2_f1p0__%s.json" % ref)))["rows"]
        pc = dict(DEFAULTS, **((cfg.get("proposed") or {}).get("plan_correlate") or {}))
        out_a, out_b = [], []
        for k in range(10):
            r = rows[str(k)]; P = np.asarray(r["P"])
            sp = preprocess.prepare(metrics.transform_cloud(src, P), cfg)
            up_src = rotation.estimate_gravity_axis(sp, cfg)
            R, _, s_AP = metrics.decompose_sim3(A @ P)
            out_a.append({"k": k, "src_angle_deg": ang(up_src, R.T @ EZ), "ref_angle_deg": ang(up_ref, EZ)})
            _, _, sA = metrics.decompose_sim3(A); _, _, sP = metrics.decompose_sim3(P); s_true = sA / sP
            sc = [metrics.decompose_sim3(np.asarray(T))[2] for T in r["diag"]["gen"]]
            out_b.append({"k": k, "s_true": float(s_true), "in_grid": bool(pc["scale_lo"] <= s_true <= pc["scale_hi"]),
                          "gen_scale_ratio": [float(x / s_true) for x in sc]})
        res["a"][ref] = out_a; res["b"][ref] = {"grid": [pc["scale_lo"], pc["scale_hi"], pc["scale_n"]], "trials": out_b}
        # 参照の法線
        n = dp.normals; lab = dp.labels; c = {}
        for name in ("floor", "ceiling"):
            m = lab == NAME_TO_ID[name]
            if m.any():
                t = np.degrees(np.arccos(np.clip(np.abs(n[m] @ EZ), 0, 1))); c[name] = q(t)
        res["c"]["ref_" + ref] = c
    cfg = yaml.safe_load(open("Registration/configs/conslam/b2_f1p0__S1_small.yaml"))
    sp = preprocess.prepare(io_utils.load_source_cloud(cfg), cfg)
    up_true = A[:3, :3].T @ EZ
    c = {}
    for name in ("floor", "ceiling", "wall"):
        m = sp.labels == NAME_TO_ID[name]
        t = np.degrees(np.arccos(np.clip(np.abs(sp.normals[m] @ up_true / np.linalg.norm(up_true)), 0, 1)))
        c[name] = dict(q(t), frac_within_10deg=float((t < 10).mean()), frac_over_80deg=float((t > 80).mean()))
    res["c"]["input_b2_f1p0"] = c
    json.dump(res, open(os.path.expanduser("~/rizu/r63_data/diag.json"), "w"), indent=1)
    for ref in REFS:
        a = res["a"][ref]; b = res["b"][ref]
        print(ref, "src angle", [round(x["src_angle_deg"], 1) for x in a], "ref angle", round(a[0]["ref_angle_deg"], 2))
        allr = [x for t in b["trials"] for x in t["gen_scale_ratio"]]
        print("   s_true in grid", sum(t["in_grid"] for t in b["trials"]), "/10", "gen scale ratio", q(allr))
    print(json.dumps(res["c"], indent=1)[:2500])


if __name__ == "__main__":
    main()
