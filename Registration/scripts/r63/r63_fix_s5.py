"""R63 §8-3：直した入力・参照での S5（条件 B・条件 A、完全・b-2・f 1p0 の 40 試行）の記録の集計（sni-slam の環境）。**回す前に commit する。位置合わせは回さない。**

- 記録：条件 B は `~/rizu/r63_data/results_fix/full/b2_f1p0__<参照>.json`（P は摂動そのもの）、
  条件 A は `~/rizu/r63_data/resultsA_fix/full/b2_f1p0__<参照>.json`（P は P_eff）。正解は G = A · P⁻¹
- 誤差は実行の時点と同じ `per_trial(T, P, A, Ω_b2)`（Ω は `r63fix_omega.npz`）。縮尺の比 = (T P の縮尺) / (A の縮尺)、傾き = `per_trial` の tilt
- 成分：Ω の点 x について d = (T P) x − A x の鉛直 d_z と水平 (d_x, d_y)。rms_z・rms_xy（rms_z² + rms_xy² = e_Ω²）と d_z の平均
- 段階：gen・short・after_icp の候補のうち E_Ω 最小の候補（b）と、基準内（e_R < 45°・e_Ω < 1.0 m）の候補があるか。全候補の d_z の平均と縮尺の比
出力：`~/rizu/r63_data/fix_s5_<B|A>.json`
使い方：python r63_fix_s5.py B ／ A
"""
import json, os, sys
import numpy as np

REPO = os.path.expanduser("~/rizu/SNI-SLAM")
for p in ("Registration", "Registration/scripts"):
    sys.path.insert(0, os.path.join(REPO, p))
os.chdir(REPO)
from regbim import metrics                                     # noqa: E402
from r41_rescore import per_trial                              # noqa: E402

CO = os.path.expanduser("~/rizu/conslam/out")
REFS = ["S1_small", "S1_large", "S3_small", "S3_large"]
A = np.array(json.load(open(os.path.join(CO, "extra.json")))["b_normalized"]["A"]); SA = metrics.decompose_sim3(A)[2]
OM = np.load(os.path.join(CO, "r63fix_omega.npz"))["omega_b2"]


def app(T, X):
    T = np.asarray(T, dtype=np.float64); return X @ T[:3, :3].T + T[:3, 3]


def ev(T, P):
    e = per_trial(T, P, A, OM, None)
    out = {"v": bool(e["v"]), "eO": float(e["eO"]), "eR": float(e["eR"]), "tilt": float(e["tilt"]), "yaw": float(e["yaw"])}
    out["ok"] = bool(out["v"] and out["eR"] < 45 and out["eO"] < 1.0)
    if out["v"]:
        Tc = np.asarray(T) @ P; d = app(Tc, OM) - app(A, OM)
        out.update(scale_ratio=float(metrics.decompose_sim3(Tc)[2] / SA), rms_z=float(np.sqrt((d[:, 2] ** 2).mean())),
                   rms_xy=float(np.sqrt((d[:, :2] ** 2).sum(1).mean())), mean_dz=float(d[:, 2].mean()))
    return out


def main():
    cond = sys.argv[1]; RD = os.path.expanduser("~/rizu/r63_data/" + {"B": "results_fix", "A": "resultsA_fix"}[cond])
    res = {"cond": cond, "trials": {}}
    for ref in REFS:
        rows = json.load(open(os.path.join(RD, "full", "b2_f1p0__%s.json" % ref)))["rows"]; out = []
        for k in range(10):
            r = rows[str(k)]; P = np.asarray(r["P"]); d = r["diag"]
            t = {"k": k, "final": ev(r["T_est"], P), "time_s": r["time_s"]}
            t["chk"] = abs(t["final"]["eO"] - r["run_eO"])
            for st, Ts in (("gen", d["gen"]), ("short", d["short"]), ("after", d["after_icp"])):
                es = [ev(T, P) for T in Ts]; i = int(np.argmin([e["eO"] for e in es]))
                t["b_" + st] = dict(es[i], idx=i); t["any_ok_" + st] = any(e["ok"] for e in es); t["n_" + st] = len(es)
                t["all_" + st] = [{x: e.get(x) for x in ("eO", "eR", "tilt", "scale_ratio", "mean_dz", "rms_xy")} for e in es]
            out.append(t)
        res["trials"][ref] = out
    json.dump(res, open(os.path.expanduser("~/rizu/r63_data/fix_s5_%s.json" % cond), "w"), indent=1)
    print("check max", max(t["chk"] for v in res["trials"].values() for t in v))
    for ref in REFS:
        T = res["trials"][ref]; F = [t["final"] for t in T]
        okr = [f for f in F if f["eR"] < 45]
        print(ref, "ok %d eR<45 %d" % (sum(f["ok"] for f in F), len(okr)),
              "eO med %.2f [%.2f %.2f]" % tuple(np.percentile([f["eO"] for f in F], [50, 25, 75])),
              "eR", sorted(round(f["eR"], 1) for f in F), "tilt med %.1f" % np.median([f["tilt"] for f in F]),
              "scale_ratio", sorted(round(f["scale_ratio"], 3) for f in F),
              "rms_z med %.2f" % np.median([f["rms_z"] for f in F]), "rms_xy(eR<45)", sorted(round(f["rms_xy"], 2) for f in okr),
              "b gen/short/after med %.2f %.2f %.2f" % tuple(np.median([t["b_" + s]["eO"] for t in T]) for s in ("gen", "short", "after")),
              "b min %.2f %.2f %.2f" % tuple(min(t["b_" + s]["eO"] for t in T) for s in ("gen", "short", "after")),
              "any_ok", [sum(t["any_ok_" + s] for t in T) for s in ("gen", "short", "after")], "time med %.0f" % np.median([t["time_s"] for t in T]))
    for st in ("gen", "short", "after"):
        al = [e for v in res["trials"].values() for t in v for e in t["all_" + st]]
        dz = [e["mean_dz"] for e in al if e["mean_dz"] is not None]; sr = [e["scale_ratio"] for e in al if e["scale_ratio"] is not None]
        ti = [e["tilt"] for e in al if np.isfinite(e["tilt"])]
        print(st, "n", len(al), "mean_dz med %.3f p5 %.3f p95 %.3f |dz|<0.5 %.3f" % (np.median(dz), np.percentile(dz, 5), np.percentile(dz, 95), np.mean(np.abs(dz) < 0.5)),
              "scale_ratio med %.3f p5 %.3f p95 %.3f in[0.95,1.05] %.3f" % (np.median(sr), np.percentile(sr, 5), np.percentile(sr, 95), np.mean((np.array(sr) > 0.95) & (np.array(sr) < 1.05))),
              "tilt med %.1f <10 %.3f" % (np.median(ti), np.mean(np.array(ti) < 10)))


if __name__ == "__main__":
    main()
