"""R61 §3-2・§3-3：同じ入力での再実行と、摂動を成分に分けた実行（sni-slam の環境、ブランチ `exp/r61`）。**回す前に commit する。**

対象：R49 の完全な方式・実データ・f 1.0（`r49_run.REAL` の 8 条件 = 4 スキャン × E2・E3）× k 0〜9 = 80 試行。
1 試行は `r49_run.run_one("real", target, "full", k, P, timed)`（R49 と同じ関数・同じ config・同じ読み込み）。

rerun（§3-2）：P は R49 と同じ（`r49_run.perturbations`）
- serial：R49 の `serial` と同じ設定。1 プロセスで 80 試行を順に、`timed=True`（`regbim.instrument` で記録。計算には触れない）。
  スレッドは呼び出し側で固定する（`taskset -c <8 コア>`、OMP・OpenBLAS・MKL = 8）。3 回を別々の 8 コア（0-7・8-15・16-23）で同時に回す
- parallel：`multiprocessing.Pool(10)`、`timed=False`（R49 の `parallel` と同じ）。スレッドの環境変数は付けない
- 出力：`~/rizu/r61_data/rerun/<serial|parallel>_r<1..3>/<target>.json`（R49 と同じ形の rows と、実行の環境 env）

comp（§3-3）：P_k を成分に分ける。座標は R55 と同じ RG（SLAM の y 上 → z 上の固定の回転、`r55_common.RG`）で、Pz = RG P RGᵀ = (R, t, s)
- R の swing-twist 分解（twist 軸 = e_z）：q = q_swing · q_twist、q_twist = normalize(w, 0, 0, z)。R = R_swing · R_twist
- P_yaw = RGᵀ · [R_twist, 0, 1] · RG（ヨーの回転だけ）
- P_xy = RGᵀ · [I, (t_x, t_y, 0), 1] · RG（水平の並進だけ）
- P_tilt = RGᵀ · [R_swing, (0, 0, t_z), 1] · RG（傾きと鉛直の並進だけ）
- P_scale = RGᵀ · [I, 0, s] · RG（縮尺だけ。原点まわり）
- none = 単位行列（摂動なし）
- 出力：`~/rizu/r61_data/comp/<yaw|xy|tilt|scale|none>/<target>.json`（rows に R49 と同じ欄と、前処理の後の点の数 `n_prep`（クラスごと））
comp34（§3-4、§8-2 の決定、2026-11-21 追加）：10 並列。出力 `~/rizu/r61_data/comp34/<変種>/<target>.json`
- xy_min：P_xy、`proposed.plan_correlate.src_grid_origin: min`（source の平面図の格子の原点を点の範囲の最小値に。`plan_correlate.horizontal_candidates`）
- none_min：摂動なし、同じスイッチ（xy_min の比べる相手）
- yaw_vframe：P_yaw、`preprocess.src_voxel_frame` = P_yaw⁻¹（source を元の座標に戻して間引き、P_yaw で戻す。`preprocess.prepare` の frame）
- none_check：摂動なし、スイッチなし（スイッチを足した後のコードで、既定の計算が §3-3 の none の記録とビット一致するかの確認）
- 1 試行は `run_one(timed=False)` と同じ手順（`clouds`・`config("real", target, "full")`・`proposed`・diag）に、上の config の変更だけを足したもの
使い方：python r61_run.py serial <r> ／ parallel <r> [並列数] ／ comp <成分,...> [並列数] ／ comp34 <変種,...> [並列数]
"""
import json, os, sys, time
from multiprocessing import Pool
import numpy as np

REPO = os.path.expanduser("~/rizu/SNI-SLAM")
for p in ("Registration", "Registration/scripts", "Registration/scripts/r49", "Registration/scripts/r55"):
    sys.path.insert(0, os.path.join(REPO, p))
os.chdir(REPO)
import r49_run as RR                                            # noqa: E402
from regbim import metrics, preprocess                          # noqa: E402
from failure_decomposition import provenance                    # noqa: E402

OUT = os.path.expanduser("~/rizu/r61_data")
RG = np.array([[1.0, 0, 0, 0], [0, 0, -1.0, 0], [0, 1.0, 0, 0], [0, 0, 0, 1.0]])   # r55_common.RG と同じ
COMPS = ("yaw", "xy", "tilt", "scale", "none")


def env():
    e = {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
    e["affinity"] = sorted(os.sched_getaffinity(0)); e["cpu_count"] = os.cpu_count(); return e


def quat(R):
    from scipy.spatial.transform import Rotation
    x, y, z, w = Rotation.from_matrix(R).as_quat(); return np.array([w, x, y, z])


def qmat(q):
    from scipy.spatial.transform import Rotation
    w, x, y, z = q; return Rotation.from_quat([x, y, z, w]).as_matrix()


def qmul(a, b):
    w1, x1, y1, z1 = a; w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def components(P):
    R, t, s = metrics.decompose_sim3(RG @ P @ RG.T)
    q = quat(R); qt = np.array([q[0], 0.0, 0.0, q[3]]); qt /= np.linalg.norm(qt)
    qs = qmul(q, np.array([qt[0], -qt[1], -qt[2], -qt[3]]))
    Rt, Rs = qmat(qt), qmat(qs)
    back = lambda M: RG.T @ M @ RG
    c = {"yaw": back(metrics.make_sim3(Rt, np.zeros(3), 1.0)), "xy": back(metrics.make_sim3(np.eye(3), np.array([t[0], t[1], 0.0]), 1.0)),
         "tilt": back(metrics.make_sim3(Rs, np.array([0.0, 0.0, t[2]]), 1.0)), "scale": back(metrics.make_sim3(np.eye(3), np.zeros(3), s)),
         "none": np.eye(4)}
    chk = float(np.abs(Rs @ Rt - R).max())
    return c, chk


def n_prep(target, P, frame=None):
    src, dst, _ = RR.clouds("real", target); cfg = RR.config("real", target, "full")
    sp = preprocess.prepare(metrics.transform_cloud(src, P), cfg, frame=frame)
    return {"total": int(len(sp.points)), "by_class": {int(k): int(v) for k, v in zip(*np.unique(sp.labels, return_counts=True))}}


def path(kind, name, target):
    return os.path.join(OUT, kind, name, target + ".json")


def load(p):
    return json.load(open(p)) if os.path.exists(p) else {"provenance": provenance(), "env": env(), "rows": {}}


def save(p, d):
    os.makedirs(os.path.dirname(p), exist_ok=True); json.dump(d, open(p, "w"))


def serial(r):
    t0 = time.time()
    for target in RR.REAL:
        Ps = RR.perturbations("real", target); p = path("rerun", "serial_r%s" % r, target); d = load(p)
        for k in range(10):
            if str(k) in d["rows"]:
                continue
            d["rows"][str(k)] = RR.run_one("real", target, "full", k, Ps[k], True); save(p, d)
            print("serial", r, target, k, "%.0fs" % d["rows"][str(k)]["time_s"], "elapsed %.0fs" % (time.time() - t0), flush=True)


def pjob(a):
    kind, name, target, k = a
    P0 = RR.perturbations("real", target)[k]
    if kind == "rerun":
        return a, RR.run_one("real", target, "full", k, P0, False)
    c, chk = components(P0); P = c[name]
    r = RR.run_one("real", target, "full", k, P, False)
    r["n_prep"] = n_prep(target, P); r["P_orig"] = P0.tolist(); r["decomp_check"] = chk
    return a, r


VARIANTS = {"xy_min": "xy", "none_min": "none", "yaw_vframe": "yaw", "none_check": "none"}


def job34(a):
    import copy
    from regbim.methods import get_method
    _, name, target, k = a
    P0 = RR.perturbations("real", target)[k]; c, _ = components(P0); P = c[VARIANTS[name]]
    src, dst, _ = RR.clouds("real", target); cfg = copy.deepcopy(RR.config("real", target, "full")); frame = None
    if name in ("xy_min", "none_min"):
        cfg.setdefault("proposed", {}).setdefault("plan_correlate", {})["src_grid_origin"] = "min"
    if name == "yaw_vframe":
        frame = metrics.invert_sim3(P); cfg["preprocess"] = dict(cfg["preprocess"], src_voxel_frame=frame.tolist())
    meth = get_method("proposed"); t0 = time.perf_counter()
    T = meth.register(metrics.transform_cloud(src, P), dst, cfg); dt = time.perf_counter() - t0
    pd, yd = meth.last_plan_diag, meth.last_yaw_diag
    diag = {"gen": [x["T_gen"] for x in pd["stage_candidates"]], "short": [x["T_short"] for x in pd["stage_candidates"]],
            "after_icp": yd["candidate_T"], "scores": yd["candidate_scores"]}
    return a, {"T_est": np.asarray(T, dtype=np.float64).tolist(), "P": P.tolist(), "time_s": dt, "diag": diag,
               "n_prep": n_prep(target, P, frame), "P_orig": P0.tolist()}


def parallel(jobs, npar):
    have, todo = {}, []
    for a in jobs:
        p = path(a[0], a[1], a[2])
        if p not in have:
            have[p] = load(p)
        if str(a[3]) not in have[p]["rows"]:
            todo.append(a)
    todo.sort(key=lambda a: (a[2], a[1], a[3])); print("todo", len(todo), flush=True); t0 = time.time()
    with Pool(npar) as pool:
        for i, (a, r) in enumerate(pool.imap_unordered(job34 if todo and todo[0][0] == "comp34" else pjob, todo, chunksize=1)):
            p = path(a[0], a[1], a[2]); have[p]["rows"][str(a[3])] = r; save(p, have[p])
            print("done", i + 1, len(todo), a, "%.0fs" % r["time_s"], "elapsed %.0fs" % (time.time() - t0), flush=True)


def main():
    mode = sys.argv[1]
    if mode == "serial":
        serial(sys.argv[2])
    elif mode == "parallel":
        r = sys.argv[2]; npar = int(sys.argv[3]) if len(sys.argv) > 3 else 10
        parallel([("rerun", "parallel_r%s" % r, t, k) for t in RR.REAL for k in range(10)], npar)
    elif mode == "comp":
        cs = sys.argv[2].split(","); npar = int(sys.argv[3]) if len(sys.argv) > 3 else 10
        assert all(c in COMPS for c in cs)
        parallel([("comp", c, t, k) for c in cs for t in RR.REAL for k in range(10)], npar)
    elif mode == "comp34":
        vs = sys.argv[2].split(","); npar = int(sys.argv[3]) if len(sys.argv) > 3 else 10
        assert all(v in VARIANTS for v in vs)
        ks = range(int(sys.argv[4])) if len(sys.argv) > 4 else range(10)
        tg = RR.REAL if len(sys.argv) <= 5 else sys.argv[5].split(",")
        parallel([("comp34", v, t, k) for v in vs for t in tg for k in ks], npar)
    elif mode == "decomp_check":
        for t in RR.REAL:
            for k, P in enumerate(RR.perturbations("real", t)):
                c, chk = components(P); print(t, k, "swing*twist-R %.2e" % chk)


if __name__ == "__main__":
    main()
