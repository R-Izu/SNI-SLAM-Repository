"""R57 §3-3：座標がそろっているかの確認（sni-slam の環境）。**回す前に commit する。位置合わせは回さない。**

- 地上レーザー（TLS）：`sequence2/Copy of groundtruth_scan.ply`。ヘッダは 163,007,486 点だが、ファイルが途中で切れていて、
  完全に読めるのは先頭の 159,383,534 点（1 点 19 バイト）。その範囲を memmap で読み、seed 0 で 2,000,000 点を一様に抜く
- IFC：`out/conslam_ifc_dense5m.npz`（5,000,000 点、距離の道具）。参照の作り方の 20 万点（`conslam_ifc_all.npz`）でも同じ距離を出す
- 距離：(1) TLS の標本 → IFC の最近点、(2) TLS の xy・z の範囲に入る IFC の点 → TLS の標本の最近点。中央値・90%・95%。
  道具の分解能として、IFC の点どうし・TLS の標本どうしの最近点の距離の中央値も出す
- GT の姿勢で LiDAR の走査を置く：S2 の GT（`S2_All_LiDAR_poses_v5_denser_T31.tum`）の行のうち、`lidar-pcd/` に同じ時刻（μs）の走査がある行から、
  等間隔に 10 本。TUM の行（t, x, y, z, qx, qy, qz, qw）から作る 4×4 を、走査の点にそのままかける。置いた点 → TLS・IFC の最近点の距離
出力：`~/rizu/conslam/out/coords.json`、図 `coords_top.png`・`coords_z.png`
"""
import glob, json, os
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

D = "/mnt/d/rizu/SNI-SLAM/data/ConSLAM"
OUT = os.path.expanduser("~/rizu/conslam/out")
W = os.path.expanduser("~/rizu/conslam/work")


def q(d):
    d = np.asarray(d)
    return {"n": int(len(d)), "median": float(np.median(d)), "p90": float(np.percentile(d, 90)), "p95": float(np.percentile(d, 95))}


def read_tls(n_sample=2000000):
    p = os.path.join(D, "sequence2", "Copy of groundtruth_scan.ply")
    h = open(p, "rb").read(2000); e = h.index(b"end_header\n") + len(b"end_header\n")
    n = (os.path.getsize(p) - e) // 19
    dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1"), ("i", "<f4")])
    mm = np.memmap(p, dtype=dt, mode="r", offset=e, shape=(n,))
    lo = np.full(3, np.inf); hi = np.full(3, -np.inf)
    for s in range(0, n, 10000000):
        c = mm[s:s + 10000000]; xyz = np.stack([c["x"], c["y"], c["z"]], 1)
        lo = np.minimum(lo, xyz.min(0)); hi = np.maximum(hi, xyz.max(0))
    idx = np.sort(np.random.default_rng(0).choice(n, n_sample, replace=False))
    c = mm[idx]
    return np.stack([c["x"], c["y"], c["z"]], 1).astype(np.float64), {"n_readable": int(n), "bbox_min": lo.tolist(), "bbox_max": hi.tolist()}


def read_pcd(p):
    with open(p, "rb") as f:
        while True:
            l = f.readline()
            if l.startswith(b"POINTS"):
                n = int(l.split()[1])
            if l.startswith(b"DATA"):
                break
        return np.frombuffer(f.read(n * 12), dtype="<f4").reshape(n, 3).astype(np.float64)


def main():
    res = {}
    tls, res["tls"] = read_tls()
    dense = np.load(os.path.join(OUT, "conslam_ifc_dense5m.npz")); ifc = dense["points"].astype(np.float64)
    ref200 = np.load(os.path.join(OUT, "conslam_ifc_all.npz"))["points"]
    res["ifc_dense"] = {"n": int(len(ifc)), "bbox_min": ifc.min(0).tolist(), "bbox_max": ifc.max(0).tolist()}
    t_ifc, t_tls, t_200 = cKDTree(ifc), cKDTree(tls), cKDTree(ref200)
    rng = np.random.default_rng(0)
    res["resolution"] = {"ifc_dense_nn_median": float(np.median(t_ifc.query(ifc[rng.choice(len(ifc), 200000, replace=False)], k=2)[0][:, 1])),
                         "ifc_200k_nn_median": float(np.median(t_200.query(ref200, k=2)[0][:, 1])),
                         "tls_sample_nn_median": float(np.median(t_tls.query(tls[rng.choice(len(tls), 200000, replace=False)], k=2)[0][:, 1]))}
    res["tls_to_ifc_dense"] = q(t_ifc.query(tls, k=1)[0])
    res["tls_to_ifc_200k"] = q(t_200.query(tls, k=1)[0])
    lo, hi = tls.min(0), tls.max(0)
    m = np.all((ifc >= lo) & (ifc <= hi), axis=1)
    res["ifc_dense_in_tls_box_to_tls"] = q(t_tls.query(ifc[m], k=1)[0])
    # GT で置いた LiDAR の走査
    a = np.loadtxt(os.path.join(D, "m1743877", "S2_All_LiDAR_poses_v5_denser_T31.tum"))
    pcd = {int(os.path.splitext(os.path.basename(p))[0]): p for p in glob.glob(os.path.join(W, "lidar-pcd", "*.pcd"))}
    rows = [i for i in range(len(a)) if int(round(a[i, 0] * 1e6)) in pcd]
    pick = [rows[int(round(x))] for x in np.linspace(0, len(rows) - 1, 10)]
    scans = []
    for i in pick:
        T = np.eye(4); T[:3, :3] = Rotation.from_quat(a[i, 4:8]).as_matrix(); T[:3, 3] = a[i, 1:4]
        x = read_pcd(pcd[int(round(a[i, 0] * 1e6))]); x = x[np.isfinite(x).all(1)]
        y = x @ T[:3, :3].T + T[:3, 3]
        scans.append({"row": int(i), "t": float(a[i, 0]), "n": int(len(y)), "to_tls": q(t_tls.query(y, k=1)[0]), "to_ifc_dense": q(t_ifc.query(y, k=1)[0])})
    res["gt_scans"] = {"n_rows_with_scan": len(rows), "picked": scans,
                       "pooled_to_tls_median_of_medians": float(np.median([s["to_tls"]["median"] for s in scans])),
                       "pooled_to_ifc_median_of_medians": float(np.median([s["to_ifc_dense"]["median"] for s in scans]))}
    json.dump(res, open(os.path.join(OUT, "coords.json"), "w"), indent=1)
    # 図
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(14, 8))
    a1 = ifc[rng.choice(len(ifc), 300000, replace=False)]; b1 = tls[rng.choice(len(tls), 300000, replace=False)]
    ax.scatter(a1[:, 0], a1[:, 1], s=0.2, c="#0072B2", label="IFC (dense, 300k shown)")
    ax.scatter(b1[:, 0], b1[:, 1], s=0.2, c="#E69F00", alpha=0.5, label="TLS seq2 (300k shown)")
    ax.plot(a[:, 1], a[:, 2], "k-", lw=0.8, label="S2 GT poses")
    ax.set_aspect("equal"); ax.legend(markerscale=20); ax.set_title("R57 top view (x-y)"); fig.tight_layout(); fig.savefig(os.path.join(OUT, "coords_top.png"), dpi=110); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(a1[:, 2], bins=200, alpha=0.6, label="IFC z", color="#0072B2"); ax.hist(b1[:, 2], bins=200, alpha=0.6, label="TLS z", color="#E69F00")
    ax.axvspan(a[:, 3].min(), a[:, 3].max(), color="k", alpha=0.2, label="GT pose z range"); ax.legend(); ax.set_xlabel("z [m]")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "coords_z.png"), dpi=110); plt.close(fig)
    print(json.dumps({k: v for k, v in res.items() if k != "gt_scans"}, indent=1)); print("gt_scans", json.dumps(res["gt_scans"])[:1500])


if __name__ == "__main__":
    main()
