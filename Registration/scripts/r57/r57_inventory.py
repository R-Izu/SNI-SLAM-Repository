"""R57 §3-1：系列 2 の目録と、GT 姿勢（mediaTUM）と ConSLAM の normalized の関係（sni-slam の環境）。**回す前に commit する。**
出力：`~/rizu/conslam/out/inventory.json`
"""
import glob, json, os, struct, zlib
import numpy as np
from scipy.spatial.transform import Rotation

S = "/mnt/d/rizu/SNI-SLAM/data/ConSLAM/sequence2"
W = os.path.expanduser("~/rizu/conslam/work")
B = "/mnt/d/rizu/SNI-SLAM/data/ConSLAM/m1743877"
OUT = os.path.expanduser("~/rizu/conslam/out")


def stamps(names):
    t = sorted(int(os.path.splitext(os.path.basename(n))[0]) for n in names if os.path.splitext(n)[1] in (".png", ".pcd", ".txt") and os.path.basename(n)[0].isdigit())
    t = np.asarray(t, dtype=np.int64) / 1e6
    d = np.diff(t)
    return {"n": int(len(t)), "t_first": float(t[0]), "t_last": float(t[-1]), "duration_s": float(t[-1] - t[0]),
            "dt_median_s": float(np.median(d)), "dt_p5_s": float(np.percentile(d, 5)), "dt_p95_s": float(np.percentile(d, 95)), "dt_max_s": float(d.max())}, t


def png_size(zpath, member):
    import zipfile
    with zipfile.ZipFile(zpath) as z:
        with z.open(member) as f:
            h = f.read(33)
    return struct.unpack(">II", h[16:24])


def load_tum(p):
    a = np.loadtxt(p)
    T = np.tile(np.eye(4), (len(a), 1, 1)); T[:, :3, :3] = Rotation.from_quat(a[:, 4:8]).as_matrix(); T[:, :3, 3] = a[:, 1:4]
    return a[:, 0], T


def main():
    os.makedirs(OUT, exist_ok=True)
    inv = {}
    rgb = [l.strip() for l in open(os.path.join(W, "rgb_list.txt"))]; nir = [l.strip() for l in open(os.path.join(W, "nir_list.txt"))]
    inv["rgb"], t_rgb = stamps([n for n in rgb if n.count("/") == 1])
    inv["rgb"]["other_entries"] = [n for n in rgb if n.count("/") != 1 or not n.endswith(".png")]
    inv["rgb"]["png_size_wh"] = png_size(os.path.join(S, "Copy of rgb.zip"), [n for n in rgb if n.endswith(".png")][0])
    inv["nir"], _ = stamps([n for n in nir if n.endswith(".png")])
    inv["nir"]["png_size_wh"] = png_size(os.path.join(S, "Copy of nir.zip"), [n for n in nir if n.endswith(".png")][0])
    pcds = sorted(glob.glob(os.path.join(W, "lidar-pcd", "*.pcd")))
    inv["lidar"], t_lid = stamps(pcds)
    npts = []
    for p in pcds[:: max(1, len(pcds) // 50)]:
        with open(p, "rb") as f:
            for line in f:
                if line.startswith(b"POINTS"):
                    npts.append(int(line.split()[1])); break
    inv["lidar"]["points_per_scan_sampled"] = {"n_files_sampled": len(npts), "median": float(np.median(npts)), "min": int(min(npts)), "max": int(max(npts))}
    nz = sorted(glob.glob(os.path.join(W, "normalized", "*.txt")))
    inv["normalized"], t_norm = stamps(nz)
    Tn = np.stack([np.loadtxt(p) for p in nz])
    inv["normalized"]["first_is_identity"] = bool(np.allclose(Tn[0], np.eye(4)))
    inv["normalized"]["n_match_lidar_stamp"] = int(np.isin(np.round(t_norm * 1e6).astype(np.int64), np.round(t_lid * 1e6).astype(np.int64)).sum())
    # GT（mediaTUM）
    inv["gt"] = {}
    for p in sorted(glob.glob(os.path.join(B, "*.tum"))):
        t, T = load_tum(p)
        d = np.diff(t)
        inv["gt"][os.path.basename(p)] = {"rows": int(len(t)), "t_first": float(t[0]), "t_last": float(t[-1]), "dt_median_s": float(np.median(d)),
                                          "xyz_min": T[:, :3, 3].min(0).tolist(), "xyz_max": T[:, :3, 3].max(0).tolist(),
                                          "path_length_m": float(np.linalg.norm(np.diff(T[:, :3, 3], axis=0), axis=1).sum())}
    # S2 の GT と normalized の対応（同じ時刻の行）と、1 つの剛体 A で GT = A · N と書けるか
    t_gt, T_gt = load_tum(os.path.join(B, "S2_All_LiDAR_poses_v5_denser_T31.tum"))
    key_gt = {int(round(x * 1e6)): i for i, x in enumerate(t_gt)}
    pairs = [(key_gt[int(round(x * 1e6))], j) for j, x in enumerate(t_norm) if int(round(x * 1e6)) in key_gt]
    near = [int(np.min(np.abs(t_gt - x)) < 1e-3) for x in t_norm]
    rel = {"n_gt": int(len(t_gt)), "n_norm": int(len(t_norm)), "n_exact_stamp_match": len(pairs), "n_within_1ms": int(sum(near)),
           "gt_stamps_in_lidar_stamps": int(np.isin(np.round(t_gt * 1e6).astype(np.int64), np.round(t_lid * 1e6).astype(np.int64)).sum())}
    if pairs:
        As = np.stack([T_gt[i] @ np.linalg.inv(Tn[j]) for i, j in pairs])
        A0 = As[0]
        dR = [float(np.degrees(np.arccos(np.clip((np.trace(A0[:3, :3].T @ A[:3, :3]) - 1) / 2, -1, 1)))) for A in As]
        dt = [float(np.linalg.norm(A[:3, 3] - A0[:3, 3])) for A in As]
        rel.update(A_first=A0.tolist(), A_spread_rot_deg={"median": float(np.median(dR)), "max": float(max(dR))},
                   A_spread_trans_m={"median": float(np.median(dt)), "max": float(max(dt))})
    inv["gt_vs_normalized_S2"] = rel
    # RGB と LiDAR の時刻の対応
    j = np.searchsorted(t_lid, t_rgb).clip(1, len(t_lid) - 1)
    dmin = np.minimum(np.abs(t_lid[j] - t_rgb), np.abs(t_lid[j - 1] - t_rgb))
    inv["rgb_to_nearest_lidar_s"] = {"median": float(np.median(dmin)), "p95": float(np.percentile(dmin, 95)), "max": float(dmin.max())}
    json.dump(inv, open(os.path.join(OUT, "inventory.json"), "w"), indent=1, default=float)
    print(json.dumps(inv, indent=1, default=float)[:6000])


if __name__ == "__main__":
    main()
