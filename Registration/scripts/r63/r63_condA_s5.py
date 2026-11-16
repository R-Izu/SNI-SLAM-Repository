"""R63 §7-2 の 3 の S5（条件 A・完全・b-2・f 1p0 の 40 試行）の記録の集計（sni-slam の環境）。**回す前に commit する。位置合わせは回さない。**

- 誤差は実行の時点と同じ `per_trial(T, P_eff, A, Ω_b2)`。e_Ω は Ω の点の位置の差の 2 乗平均の平方根（`criterion_verdict.d_omega`）
- 成分：Ω の点 x について d = (T P_eff) x − A x（IFC の座標）。鉛直 = d_z、水平 = (d_x, d_y)。
  それぞれの 2 乗平均の平方根（rms_z・rms_xy、rms_z² + rms_xy² = e_Ω²）と d_z の平均
- 段階：gen・short・after_icp の候補のうち E_Ω 最小の候補（b）と、基準内（e_R < 45°・e_Ω < 1.0 m）の候補があるか
- 高さ：入力 b-2 f1p0 の床・天井のラベルの点を A で IFC に置いた z と、参照の床・天井の点の z（`prepare` の後）。0.1 m の区間の数の多い順の上位 3 区間
出力：`~/rizu/r63_data/condA_s5.json`
"""
import json, os, sys
import numpy as np
import yaml

REPO = os.path.expanduser("~/rizu/SNI-SLAM")
for p in ("Registration", "Registration/scripts"):
    sys.path.insert(0, os.path.join(REPO, p))
os.chdir(REPO)
from regbim import io_utils, metrics, preprocess               # noqa: E402
from regbim.labels import NAME_TO_ID                           # noqa: E402
from r41_rescore import per_trial                              # noqa: E402

CO = os.path.expanduser("~/rizu/conslam/out"); RA = os.path.expanduser("~/rizu/r63_data/resultsA")
REFS = ["S1_small", "S1_large", "S3_small", "S3_large"]
A = np.array(json.load(open(os.path.join(CO, "extra.json")))["b_normalized"]["A"])
OM = np.load(os.path.join(CO, "r62_omega.npz"))["omega_b2"]


def app(T, X):
    T = np.asarray(T, dtype=np.float64); return X @ T[:3, :3].T + T[:3, 3]


def comp(Tc):
    d = app(Tc, OM) - app(A, OM)
    return {"rms_z": float(np.sqrt((d[:, 2] ** 2).mean())), "rms_xy": float(np.sqrt((d[:, :2] ** 2).sum(1).mean())), "mean_dz": float(d[:, 2].mean())}


def ev(T, P):
    e = per_trial(T, P, A, OM, None)
    out = {"v": bool(e["v"]), "eO": float(e["eO"]), "eR": float(e["eR"])}
    out["ok"] = bool(out["v"] and out["eR"] < 45 and out["eO"] < 1.0)
    if out["v"]:
        out.update(comp(np.asarray(T) @ P))
    return out


def peaks(z):
    b = np.floor(np.asarray(z) / 0.1).astype(int); u, c = np.unique(b, return_counts=True); o = np.argsort(-c)[:3]
    return [{"z_lo": float(u[i] * 0.1), "n": int(c[i]), "frac": float(c[i] / len(z))} for i in o]


def main():
    res = {"trials": {}, "heights": {}}
    for ref in REFS:
        rows = json.load(open(os.path.join(RA, "full", "b2_f1p0__%s.json" % ref)))["rows"]
        out = []
        for k in range(10):
            r = rows[str(k)]; P = np.asarray(r["P"]); d = r["diag"]; t = {"k": k, "final": ev(r["T_est"], P), "time_s": r["time_s"]}
            for st, Ts in (("gen", d["gen"]), ("short", d["short"]), ("after", d["after_icp"])):
                es = [ev(T, P) for T in Ts]; i = int(np.argmin([e["eO"] for e in es]))
                t["b_" + st] = dict(es[i], idx=i); t["any_ok_" + st] = any(e["ok"] for e in es); t["n_" + st] = len(es)
            out.append(t)
        res["trials"][ref] = out
        cfg = yaml.safe_load(open("Registration/configs/conslam/b2_f1p0__%s.yaml" % ref))
        dp = preprocess.prepare(io_utils.load_reference_cloud(cfg), cfg)
        res["heights"]["ref_" + ref] = {n: peaks(dp.points[dp.labels == NAME_TO_ID[n]][:, 2]) for n in ("floor", "ceiling")}
    cfg = yaml.safe_load(open("Registration/configs/conslam/b2_f1p0__S1_small.yaml"))
    sp = preprocess.prepare(io_utils.load_source_cloud(cfg), cfg); zs = app(A, sp.points)[:, 2]
    res["heights"]["input_b2_f1p0_under_A"] = {n: peaks(zs[sp.labels == NAME_TO_ID[n]]) for n in ("floor", "ceiling")}
    json.dump(res, open(os.path.expanduser("~/rizu/r63_data/condA_s5.json"), "w"), indent=1)
    for ref in REFS:
        for t in res["trials"][ref]:
            f = t["final"]
            print(ref, t["k"], "eO %.3f eR %.2f rms_z %.3f rms_xy %.3f mean_dz %.3f" % (f["eO"], f["eR"], f.get("rms_z", np.nan), f.get("rms_xy", np.nan), f.get("mean_dz", np.nan)),
                  "| b gen %.2f short %.2f after %.2f" % (t["b_gen"]["eO"], t["b_short"]["eO"], t["b_after"]["eO"]),
                  "ok", t["any_ok_gen"], t["any_ok_short"], t["any_ok_after"], "| b_after rms_z %.3f" % t["b_after"].get("rms_z", np.nan))
    print(json.dumps(res["heights"], indent=1))


if __name__ == "__main__":
    main()
