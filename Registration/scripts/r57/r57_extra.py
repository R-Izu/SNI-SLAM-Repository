"""R57 §7-2 の追加の測定 a〜e（sni-slam の環境）。**回す前に commit する。位置合わせは回さない。**

共通：§5-3 と同じ 10 本（S2 の GT のうち同じ時刻の走査がある 272 行から等間隔）。地上レーザー（TLS）の標本は seed 0 の 2,000,000 点（§5 と同じ）、
密な IFC（5,000,000 点）。距離は最近点、中央値・90%・95%
- a：GT の 4×4 の逆を走査の点にかける
- b：`normalized/` の姿勢 N を、GT と時刻（μs）が完全に一致する全 204 組の位置で Umeyama（剛体、縮尺なし）に当てて A を求め、A·N で置く。
  10 本のうち、同じ時刻の N が無いものは「N なし」と書いて飛ばす。Umeyama の残差 = |GT の位置 − A·N の位置|（中央値・最大）
- c：(c-1) GT の姿勢を時刻 t+δ（δ = −100・−50・+50・+100 ms）に補間して（位置は線形、回転は slerp、隣の GT の行の間）同じ走査にかける。
  (c-2) 同じ GT の姿勢で、前後の走査（±1 本、約 ±100 ms）を置く
- d：TLS の標本 → 密な IFC の距離を、最近点の IFC のクラスごとと、TLS の点の z の帯ごとに。帯は回す前に固定した：
  床の付近 z ∈ [9.7, 10.7)、壁の高さ [10.7, 13.1)、天井の付近 [13.1, 14.1)、それ以外（§5-2 の z の山 10.2 m・13.6 m の ±0.5 m を床・天井とした）
- e：GT で置いた 10 本の走査 → TLS の距離を、走査の点のセンサからの距離（0〜5・5〜10・10 m 以上）ごとに。TLS の標本を seed 1 の 20,000,000 点にしたときの同じ距離も
出力：`~/rizu/conslam/out/extra.json`
"""
import glob, json, os, sys
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r57_coords import read_tls, read_pcd, q, D, OUT, W      # noqa: E402

NAMES = {0: "background", 1: "wall", 2: "door", 3: "floor", 4: "window", 5: "ceiling"}


def T_of(row):
    T = np.eye(4); T[:3, :3] = Rotation.from_quat(row[4:8]).as_matrix(); T[:3, 3] = row[1:4]; return T


def place(T, x):
    return x @ T[:3, :3].T + T[:3, 3]


def umeyama_rigid(a, b):
    ma, mb = a.mean(0), b.mean(0); H = (a - ma).T @ (b - mb)
    U, _, Vt = np.linalg.svd(H); Dg = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ Dg @ U.T; T = np.eye(4); T[:3, :3] = R; T[:3, 3] = mb - R @ ma; return T


def main():
    res = {}
    tls, _ = read_tls(); t_tls = cKDTree(tls)
    z = np.load(os.path.join(OUT, "conslam_ifc_dense5m.npz")); ifc = z["points"].astype(np.float64); lab = z["labels"]; t_ifc = cKDTree(ifc)
    a = np.loadtxt(os.path.join(D, "m1743877", "S2_All_LiDAR_poses_v5_denser_T31.tum"))
    pcd = {int(os.path.splitext(os.path.basename(p))[0]): p for p in glob.glob(os.path.join(W, "lidar-pcd", "*.pcd"))}
    st = np.array(sorted(pcd))
    rows = [i for i in range(len(a)) if int(round(a[i, 0] * 1e6)) in pcd]
    pick = [rows[int(round(x))] for x in np.linspace(0, len(rows) - 1, 10)]
    scans = {i: read_pcd(pcd[int(round(a[i, 0] * 1e6))]) for i in pick}
    scans = {i: x[np.isfinite(x).all(1)] for i, x in scans.items()}
    def dist(y):
        return {"to_tls": q(t_tls.query(y, k=1)[0]), "to_ifc": q(t_ifc.query(y, k=1)[0])}
    # a
    res["a_inverse"] = [dict(row=i, **dist(place(np.linalg.inv(T_of(a[i])), scans[i]))) for i in pick]
    # b
    nz = {int(os.path.splitext(os.path.basename(p))[0]): np.loadtxt(p) for p in glob.glob(os.path.join(W, "normalized", "*.txt"))}
    pairs = [(i, nz[int(round(a[i, 0] * 1e6))]) for i in range(len(a)) if int(round(a[i, 0] * 1e6)) in nz]
    A = umeyama_rigid(np.array([N[:3, 3] for _, N in pairs]), np.array([a[i, 1:4] for i, _ in pairs]))
    resid = [float(np.linalg.norm((A @ N)[:3, 3] - a[i, 1:4])) for i, N in pairs]
    res["b_normalized"] = {"n_pairs": len(pairs), "A": A.tolist(), "resid_trans_m": {"median": float(np.median(resid)), "max": float(max(resid))}, "scans": []}
    for i in pick:
        k = int(round(a[i, 0] * 1e6))
        res["b_normalized"]["scans"].append(dict(row=i, **dist(place(A @ nz[k], scans[i]))) if k in nz else {"row": i, "note": "N なし"})
    # c-1
    rot = Rotation.from_quat(a[:, 4:8]); sl = Slerp(a[:, 0], rot)
    res["c1_pose_time_shift"] = {}
    for dms in (-100, -50, 50, 100):
        out = []
        for i in pick:
            t = a[i, 0] + dms / 1000.0
            if t < a[0, 0] or t > a[-1, 0]:
                out.append({"row": i, "note": "範囲外"}); continue
            T = np.eye(4); T[:3, :3] = sl([t]).as_matrix()[0]
            T[:3, 3] = [np.interp(t, a[:, 0], a[:, c]) for c in (1, 2, 3)]
            out.append(dict(row=i, **dist(place(T, scans[i]))))
        res["c1_pose_time_shift"]["%+d" % dms] = out
    # c-2
    res["c2_neighbor_scans"] = {}
    for d in (-1, 1):
        out = []
        for i in pick:
            j = int(np.searchsorted(st, int(round(a[i, 0] * 1e6)))) + d
            if j < 0 or j >= len(st):
                out.append({"row": i, "note": "範囲外"}); continue
            x = read_pcd(pcd[int(st[j])]); x = x[np.isfinite(x).all(1)]
            out.append(dict(row=i, dt_ms=float((st[j] - round(a[i, 0] * 1e6)) / 1000.0), **dist(place(T_of(a[i]), x))))
        res["c2_neighbor_scans"]["%+d" % d] = out
    # d
    dd, ii = t_ifc.query(tls, k=1)
    res["d_by_ifc_class"] = {NAMES[int(c)]: q(dd[lab[ii] == c]) for c in np.unique(lab[ii])}
    zb = {"floor_band[9.7,10.7)": (9.7, 10.7), "wall_band[10.7,13.1)": (10.7, 13.1), "ceiling_band[13.1,14.1)": (13.1, 14.1)}
    res["d_by_z_band"] = {k: q(dd[(tls[:, 2] >= lo) & (tls[:, 2] < hi)]) for k, (lo, hi) in zb.items()}
    other = ~np.any([(tls[:, 2] >= lo) & (tls[:, 2] < hi) for lo, hi in zb.values()], axis=0)
    res["d_by_z_band"]["other"] = q(dd[other])
    # e
    res["e_by_range"] = []
    for i in pick:
        x = scans[i]; r = np.linalg.norm(x, axis=1); y = place(T_of(a[i]), x); d0 = t_tls.query(y, k=1)[0]
        res["e_by_range"].append({"row": i, **{b: (q(d0[m]) if m.any() else None) for b, m in
                                               (("0-5m", r < 5), ("5-10m", (r >= 5) & (r < 10)), ("10m+", r >= 10))}})
    del t_tls
    p = os.path.join(D, "sequence2", "Copy of groundtruth_scan.ply")
    h = open(p, "rb").read(2000); e = h.index(b"end_header\n") + len(b"end_header\n"); n = (os.path.getsize(p) - e) // 19
    dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1"), ("i", "<f4")])
    mm = np.memmap(p, dtype=dt, mode="r", offset=e, shape=(n,))
    idx = np.sort(np.random.default_rng(1).choice(n, 20000000, replace=False)); c = mm[idx]
    big = np.stack([c["x"], c["y"], c["z"]], 1).astype(np.float32); t_big = cKDTree(big)
    res["e_tls20m"] = {"nn_median": float(np.median(t_big.query(big[np.random.default_rng(0).choice(len(big), 200000, replace=False)], k=2)[0][:, 1])),
                       "scans": [{"row": i, "to_tls20m": q(t_big.query(place(T_of(a[i]), scans[i]), k=1)[0])} for i in pick]}
    json.dump(res, open(os.path.join(OUT, "extra.json"), "w"), indent=1, ensure_ascii=False)
    print("done")


if __name__ == "__main__":
    main()
