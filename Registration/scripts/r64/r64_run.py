"""R64：BIM の大きさの窓で区切った入力での FPFH＋RANSAC と FGR（sni-slam の環境、ブランチ `exp/r61`）。**回す前に commit する。**

規則（指示書 §2、回す前に固定）
- 対象：実データ 4 スキャン × 参照 E2（411）× 摂動 k 0〜9 = 40 試行、f 1.0。条件 A の入力 q = P_A p（`r55_common.cond_A(P)[0]`、R55 と同じ）
- 方式：`baseline_ransac_p2l`・`baseline_fgr_p2l`（R55 と同じ。config は `r55_common.cfg_of(target, "1p0")`、パラメータは変えない）
- 窓の大きさ：参照 E2（`load_reference_cloud(cfg)` の全点）の xy の外接矩形の対角 D、L = D + 1.0 m の正方形
- 窓の置き方：入力 q の座標で xy の軸にそろえる。左下の角を x = x_min + i·L/4（i = 0, 1, …, ceil(max(x_max − x_min − L, 0) / (L/4))）、y も同じ。
  窓 = [x, x + L) × [y, y + L)。z は切らない
- 使う窓：窓の中の点が入力の点の 5% 以上の窓だけ
- 各窓：窓の中の点だけを source にして方式を回す（`cloud.subset`）。出力 T は q の座標のまま
- 窓の選び方：各窓の T で入力の全点を置き、参照の点のうち 0.3 m 以内に入力の点がある割合（参照の被覆率）。最大の窓の T を出力にする。クラスは使わない
- 診断（性能に数えない）：e_Ω が最小の窓
- 誤差：`per_trial(T, P_A, G1, Ω, Ω_BIM)`（R55 と同じ）。基準内 = v ∧ e_R < 45° ∧ e_Ω < 1.0 m
出力：`~/rizu/r64_data/<fpfh|fgr>/<target>.json`（"k" → 窓ごとの T・被覆率・誤差、選んだ窓、時間）
使い方：python r64_run.py plan ／ run [並列数]
"""
import json, math, os, sys, time
from multiprocessing import Pool
import numpy as np
from scipy.spatial import cKDTree

REPO = os.path.expanduser("~/rizu/SNI-SLAM")
for p in ("Registration", "Registration/scripts", "Registration/scripts/r49", "Registration/scripts/r55"):
    sys.path.insert(0, os.path.join(REPO, p))
os.chdir(REPO)
import r55_common as C                                          # noqa: E402
from regbim import io_utils, metrics                           # noqa: E402
from regbim.methods import get_method                          # noqa: E402
from r41_rescore import per_trial, real_criteria               # noqa: E402
from failure_decomposition import provenance                   # noqa: E402

OUT = os.path.expanduser("~/rizu/r64_data")
TARGETS = ["m3_cor_%s__E2" % s for s in "abcd"]
METH = {"fpfh": "baseline_ransac_p2l", "fgr": "baseline_fgr_p2l"}
_C = {}


def clouds(target):
    if target not in _C:
        cfg = C.cfg_of(target, "1p0"); _C.clear(); _C[target] = (io_utils.load_source_cloud(cfg), io_utils.load_reference_cloud(cfg), cfg)
    return _C[target]


def windows(q, dst):
    xy = dst.points[:, :2]; D = float(np.linalg.norm(xy.max(0) - xy.min(0))); L = D + 1.0; st = L / 4
    lo, hi = q[:, :2].min(0), q[:, :2].max(0)
    n = [int(math.ceil(max(hi[a] - lo[a] - L, 0) / st)) for a in (0, 1)]
    out = []
    for i in range(n[0] + 1):
        for j in range(n[1] + 1):
            x0, y0 = lo[0] + i * st, lo[1] + j * st
            m = (q[:, 0] >= x0) & (q[:, 0] < x0 + L) & (q[:, 1] >= y0) & (q[:, 1] < y0 + L)
            if m.mean() >= 0.05:
                out.append({"ij": [i, j], "x0y0": [float(x0), float(y0)], "frac": float(m.mean()), "mask": m})
    return out, {"D": D, "L": L, "step": st, "grid": [n[0] + 1, n[1] + 1]}


def plan():
    res = {}
    for t in TARGETS:
        src, dst, cfg = clouds(t)
        for k, P in enumerate(C.P_of(t, kk) for kk in range(10)):
            q = metrics.transform_cloud(src, C.cond_A(P)[0]).points
            w, info = windows(q, dst)
            res["%s|%d" % (t, k)] = dict(info, n_windows=len(w), n_src=int(len(q)), win_points=[int(x["mask"].sum()) for x in w])
            print(t, k, info, "windows", len(w), flush=True)
    json.dump(res, open(os.path.join(OUT, "plan.json"), "w"), indent=1)


def job(a):
    m, t, k = a
    src, dst, cfg = clouds(t)
    P = C.P_of(t, k); PA = C.cond_A(P)[0]; qc = metrics.transform_cloud(src, PA); q = qc.points
    G = real_criteria(t.split("__")[0])[1]; om = np.load("Registration/output/diag/omega.npz")[t.split("__")[0]]
    omb = np.load("Registration/output/diag/r39_omega_bim.npz")[t]
    w, info = windows(q, dst); tree_ref = dst.points; meth = get_method(METH[m])
    rows = []; t0 = time.perf_counter()
    for x in w:
        t1 = time.perf_counter()
        T = np.asarray(meth.register(qc.subset(x["mask"]), dst, cfg), dtype=np.float64); dt = time.perf_counter() - t1
        y = q @ T[:3, :3].T + T[:3, 3] if np.isfinite(T).all() else None
        cov = float((cKDTree(y).query(tree_ref, k=1)[0] <= 0.3).mean()) if y is not None else -1.0
        e = per_trial(T, PA, G, om, omb)
        rows.append({"ij": x["ij"], "frac": x["frac"], "n_points": int(x["mask"].sum()), "T": T.tolist(), "coverage": cov, "time_s": dt,
                     "v": bool(e["v"]), "eR": float(e["eR"]), "eO": float(e["eO"]), "eOB": (float(e["eOB"]) if e["eOB"] is not None else None)})
    sel = int(np.argmax([r["coverage"] for r in rows])); best = int(np.argmin([r["eO"] if r["v"] else np.inf for r in rows]))
    return a, {"P_A": PA.tolist(), "window": info, "windows": rows, "selected": sel, "closest": best, "time_s": time.perf_counter() - t0}


def run(npar):
    have, todo = {}, []
    for m in METH:
        for t in TARGETS:
            p = os.path.join(OUT, m, t + ".json"); have[(m, t)] = json.load(open(p)) if os.path.exists(p) else {"provenance": provenance(), "rows": {}}
            todo += [(m, t, k) for k in range(10) if str(k) not in have[(m, t)]["rows"]]
    todo.sort(key=lambda a: (a[1], a[0], a[2])); print("todo", len(todo), flush=True); t0 = time.time()
    with Pool(npar) as pool:
        for i, (a, r) in enumerate(pool.imap_unordered(job, todo, chunksize=1)):
            m, t, k = a; have[(m, t)]["rows"][str(k)] = r
            os.makedirs(os.path.join(OUT, m), exist_ok=True); json.dump(have[(m, t)], open(os.path.join(OUT, m, t + ".json"), "w"))
            s = r["windows"][r["selected"]]
            print("done", i + 1, len(todo), a, "win %d" % len(r["windows"]), "%.0fs" % r["time_s"], "sel eO %.2f eR %.1f cov %.3f" % (s["eO"], s["eR"], s["coverage"]),
                  "elapsed %.0fs" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    if sys.argv[1] == "plan":
        plan()
    else:
        run(int(sys.argv[2]) if len(sys.argv) > 2 else 10)
