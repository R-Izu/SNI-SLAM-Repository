"""R56：共通の調整を 2 種類（N：クラスなし、S：クラスあり）にして R55 の大域の出力にかけ直す（sni-slam の環境、ブランチ `exp/r56`）。
**回す前に commit する。大域の段は回さない。** 指示書 §3。

- 大域の出力：R55 の `~/rizu/r55_data/trials.json`（方式 prop・stsl・fpfh・fgr × 条件 A・B。STSL は k = 1。T は入力 q = P_eff p → BIM）
- 調整 N：R55 の値をそのまま使う（`T_refined`・`refined`。`r55_common.common_icp`）
- 調整 S：`regbim.semantic_icp.semantic_icp` を呼ぶ側で引数を渡すだけ（S5）。
  src = `preprocess.prepare(transform_cloud(src, T P_eff), cfg)`（意味ラベル付きの source を T で BIM の座標に置き、voxel 0.05・法線）、
  dst = `preprocess.prepare(load_reference_cloud(cfg), cfg)`。cfg は R49 の完全と同じで、`semantic_icp.with_scaling` だけ False にした写し
  （max_corr_dist 0.3・tukey_c 0.3・max_iter 50・convergence_delta 1e-5・クラスは match_classes）。`rotation_fixed=False`、`label_filter=True`（既定）、
  初期値は単位行列。結果 T_S = U T（U は剛体。T の縮尺は変えない。＊ `with_scaling=False` で T から直接始めると縮尺が 1 に置き換わるため）
- 推定なし（v = 0）の試行には、どちらの調整もかけない
- cor_d の確認（§3-4）：本手法の cor_d × E2・E3 × f 1p0 × 条件 A・B。調整 N を同じ設定で回し直して、R55 の T_refined との差を書き、
  Open3D の `correspondence_set`（収束した変換での対応）の source の点を G1 P_eff⁻¹ T⁻¹ で置いたときに、R53 の `Ref.inside`（参照の床の 0.1 m 格子を 8 近傍で 1 回膨らませ、
  床の z の最小〜天井の z の最大）の外にある割合。調整 S の当てはまり（＊）＝最後の変換で、同じクラスの参照の点が 0.3 m 未満にある source の点の割合とその距離の RMS
- 調整による動き：ΔT = T_after T⁻¹ の回転の角、Ω の重心を T P_eff で置いた位置の移動の大きさと、水平面の方位（atan2(dy, dx)、°）（＊ 並進は Ω の重心で測る）
出力：試行ごと `~/rizu/r56_data/trials.json`、cor_d `~/rizu/r56_data/cor_d.json`、確認 `Registration/output/diag/r56/check.json`
使い方：python r56_refine.py check → refine [並列数] → cor_d
"""
import copy, json, os, sys, time
from multiprocessing import Pool
import numpy as np

REPO = os.path.expanduser("~/rizu/SNI-SLAM")
for p in ("Registration", "Registration/scripts", "Registration/scripts/r49", "Registration/scripts/r52", "Registration/scripts/r53",
          "Registration/scripts/r55"):
    sys.path.insert(0, os.path.join(REPO, p))
os.chdir(REPO)
from regbim import io_utils, metrics, preprocess                # noqa: E402
from regbim.semantic_icp import semantic_icp                    # noqa: E402
from regbim.labels import NAME_TO_ID                            # noqa: E402
from failure_decomposition import provenance                    # noqa: E402
from r41_rescore import per_trial, real_criteria                # noqa: E402
import r55_common as C                                          # noqa: E402

OUT = os.path.expanduser("~/rizu/r56_data")
DIAG = "Registration/output/diag/r56"


def rigid_cfg(cfg):
    c = copy.deepcopy(cfg)
    c["semantic_icp"] = dict(c["semantic_icp"], with_scaling=False)
    return c


def refine_S(src, dst_p, cfg, T, P):
    sp = preprocess.prepare(metrics.transform_cloud(src, np.asarray(T) @ np.asarray(P)), cfg)
    t0 = time.perf_counter()
    U = semantic_icp(sp, dst_p, np.eye(4), rigid_cfg(cfg), rotation_fixed=False, label_filter=True)
    dt = time.perf_counter() - t0
    # 当てはまり（＊）
    from scipy.spatial import cKDTree
    moved = metrics.apply_sim3(U, sp.points); dmin = np.full(len(moved), np.inf)
    for c in [NAME_TO_ID[n] for n in cfg["classes"]["match_classes"]]:
        m, md = sp.labels == c, dst_p.labels == c
        if m.any() and md.any():
            dmin[m] = cKDTree(dst_p.points[md]).query(moved[m], k=1)[0]
    ok = dmin < 0.3
    return U @ np.asarray(T), dt, float(ok.mean()), float(np.sqrt((dmin[ok] ** 2).mean())) if ok.any() else None


def load_r55():
    T = json.load(open(os.path.join(C.OUT, "trials.json")))
    return T, [r for r in T["rows"] if not r["method"].startswith("stsl_run")]


def job(a):
    target, f, rows = a
    scan = target.split("__")[0]
    G = np.asarray(real_criteria(scan)[1], dtype=np.float64)
    om = np.load("Registration/output/diag/omega.npz")[scan]; omb = np.load("Registration/output/diag/r39_omega_bim.npz")[target]
    cfg = C.cfg_of(target, f); src = io_utils.load_source_cloud(cfg)
    dst_p = preprocess.prepare(io_utils.load_reference_cloud(cfg), cfg)
    out, cache = [], {}
    for r in rows:
        o = {k: r[k] for k in ("method", "cond", "target", "scan", "ref", "f", "k", "valid", "no_candidate", "prep_failed")}
        o["raw"] = r["raw"]; o["N"] = r["refined"]; o["S"] = None
        if r["valid"] and r["raw"]["v"]:
            key = (np.asarray(r["T"]).tobytes(), np.asarray(r["P_eff"]).tobytes())
            if key not in cache:
                cache[key] = refine_S(src, dst_p, cfg, r["T"], r["P_eff"])
            TS, dt, fit, rmse = cache[key]
            e = per_trial(TS, np.asarray(r["P_eff"]), G, om, omb)
            o["S"] = {x: (None if e[x] is None else float(e[x])) for x in ("v", "eR", "eO", "eOB", "q")}
            o.update(S_s=dt, S_fitness=fit, S_rmse=rmse, T_S=TS.tolist())
        o["T"] = r["T"]; o["P_eff"] = r["P_eff"]; o["T_N"] = r.get("T_refined")
        out.append(o)
    return out


def refine(npar):
    T, rows = load_r55()
    groups = {}
    for r in rows:
        groups.setdefault((r["target"], r["f"]), []).append(r)
    with Pool(npar) as pool:
        outs = pool.map(job, [(t, f, rs) for (t, f), rs in sorted(groups.items())], chunksize=1)
    os.makedirs(OUT, exist_ok=True)
    allr = [x for o in outs for x in o]
    json.dump({"provenance": provenance(), "r55_provenance": T["provenance"], "rows": allr}, open(os.path.join(OUT, "trials.json"), "w"), default=float)
    print("rows", len(allr), "S done", sum(x["S"] is not None for x in allr))


def check():
    """§3-2：本手法の 1 試行で、回転を固定した semantic_icp が通常の精緻化の結果（after_icp）と一致するか。
    R52 の f 0p6・m3_cor_b__E2・k 0 の記録（P のまま、R49 の完全な方式）。16 の短い精緻化の後の変換のどれから始めると
    after_icp[i] にビット一致するかを全組で見る（同じ cfg、`rotation_fixed=True`、`label_filter=True`。縮尺あり＝本手法の既定）"""
    target, f, k = "m3_cor_b__E2", "0p6", 0
    cfg = C.cfg_of(target, f); src = io_utils.load_source_cloud(cfg)
    r = json.load(open(os.path.join(C.R52.OUT, "f" + f, target + ".json")))["rows"][str(k)]
    P = np.asarray(r["P"]); sp = preprocess.prepare(metrics.transform_cloud(src, P), cfg)
    dp = preprocess.prepare(io_utils.load_reference_cloud(cfg), cfg)
    res = []
    for i, Ta in enumerate(r["diag"]["after_icp"]):
        best = None
        for j, Ts in enumerate(r["diag"]["short"]):
            T = semantic_icp(sp, dp, np.asarray(Ts), cfg, rotation_fixed=True, label_filter=True)
            d = float(np.abs(T - np.asarray(Ta)).max())
            if best is None or d < best[1]:
                best = (j, d)
        res.append({"after_icp": i, "best_short": best[0], "max_abs_diff": best[1]})
        print(res[-1], flush=True)
    os.makedirs(DIAG, exist_ok=True)
    json.dump({"provenance": provenance(), "trial": [target, f, k], "rows": res}, open(os.path.join(DIAG, "check.json"), "w"), indent=1)


def cor_d():
    import open3d as o3d
    from r53_distinguish import Ref
    _, rows = load_r55()
    S = {(r["method"], r["cond"], r["target"], r["f"], r["k"]): r for r in json.load(open(os.path.join(OUT, "trials.json")))["rows"]}
    out = []
    for target in ("m3_cor_d__E2", "m3_cor_d__E3"):
        cfg = C.cfg_of(target, "1p0"); src = io_utils.load_source_cloud(cfg); dst = io_utils.load_reference_cloud(cfg); ref = Ref(cfg)
        G = np.asarray(real_criteria("m3_cor_d")[1]); om = np.load("Registration/output/diag/omega.npz")["m3_cor_d"]; c0 = om.mean(0)
        for r in rows:
            if r["method"] != "prop" or r["target"] != target or r["f"] != "1p0":
                continue
            T, P = np.asarray(r["T"]), np.asarray(r["P_eff"])
            q = metrics.apply_sim3(P, src.points)
            def pc(x):
                p = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(x, dtype=np.float64))).voxel_down_sample(C.ICP["voxel"])
                p.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=C.ICP["normal_radius"], max_nn=C.ICP["normal_max_nn"]))
                return p
            s, d = pc(metrics.apply_sim3(T, q)), pc(dst.points)
            res = o3d.pipelines.registration.registration_icp(
                s, d, C.ICP["dist"], np.eye(4), o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=C.ICP["rel"], relative_rmse=C.ICP["rel"], max_iteration=C.ICP["max_iter"]))
            TN = np.asarray(res.transformation) @ T
            corr = np.asarray(res.correspondence_set)
            xs = np.asarray(s.points)[corr[:, 0]] if len(corr) else np.zeros((0, 3))
            x_gt = metrics.apply_sim3(G @ np.linalg.inv(P) @ np.linalg.inv(T), xs)
            outside = float(1.0 - ref.inside(x_gt).mean()) if len(xs) else None
            # 全点（調整の前の source の点）を G1 で置いたときの外の割合（参考）
            all_gt = metrics.apply_sim3(G @ np.linalg.inv(P) @ np.linalg.inv(T), np.asarray(s.points))
            srow = S[("prop", r["cond"], target, "1p0", r["k"])]

            def move(Ta):
                dT = np.asarray(Ta) @ np.linalg.inv(T)
                R, _, _ = metrics.decompose_sim3(dT)
                ang = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
                v = metrics.apply_sim3(np.asarray(Ta) @ P, c0[None])[0] - metrics.apply_sim3(T @ P, c0[None])[0]
                return {"rot_deg": ang, "trans_m": float(np.linalg.norm(v)), "azimuth_deg": float(np.degrees(np.arctan2(v[1], v[0]))), "dz_m": float(v[2])}
            out.append({"target": target, "cond": r["cond"], "k": r["k"],
                        "eO": {"raw": r["raw"]["eO"], "N": srow["N"]["eO"] if srow["N"] else None, "S": srow["S"]["eO"] if srow["S"] else None},
                        "eOB": {"raw": r["raw"]["eOB"], "N": srow["N"]["eOB"] if srow["N"] else None, "S": srow["S"]["eOB"] if srow["S"] else None},
                        "eR": {"raw": r["raw"]["eR"], "N": srow["N"]["eR"] if srow["N"] else None, "S": srow["S"]["eR"] if srow["S"] else None},
                        "move_N": move(TN), "move_S": move(srow["T_S"]),
                        "N_recompute_max_abs_diff_vs_r55": float(np.abs(TN - np.asarray(r["T_refined"])).max()),
                        "N_n_corr": int(len(corr)), "N_n_src": int(len(s.points)), "N_outside_frac_corr": outside,
                        "N_outside_frac_all_src": float(1.0 - ref.inside(all_gt).mean()),
                        "N_fitness": float(res.fitness), "N_rmse": float(res.inlier_rmse),
                        "N_fitness_r55": r.get("icp_fitness"), "S_fitness": srow.get("S_fitness"), "S_rmse": srow.get("S_rmse")})
            print(target, r["cond"], r["k"], out[-1]["eO"], "outside", outside, "diff", out[-1]["N_recompute_max_abs_diff_vs_r55"], flush=True)
    json.dump({"provenance": provenance(), "rows": out}, open(os.path.join(OUT, "cor_d.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    m = sys.argv[1]
    if m == "check":
        check()
    elif m == "refine":
        refine(int(sys.argv[2]) if len(sys.argv) > 2 else 12)
    else:
        cor_d()
