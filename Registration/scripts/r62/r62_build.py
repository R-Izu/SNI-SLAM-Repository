"""R62：ConSLAM の入力・参照・評価点・正解の確認・config を作る（sni-slam の環境、ブランチ `exp/r62`）。**回す前に commit する。位置合わせは回さない。**

使い方：python r62_build.py orient → build → refs → check → configs

orient（§3-1 の 2、§4 S5）
- `calib_lidar2rgb.txt` の 4×4 を M とし、LiDAR → カメラに (i) M、(ii) M⁻¹ を使う 2 通り
- カメラの座標で z > 0 の点を `cv2.projectPoints`（K = `calib_rgb.yaml` の `camera_matrix`、歪み = `distortion_coefficients`、回転・並進 0）で
  元の画像（2,064 × 1,544、歪みはそのまま）に投影し、画像の中に入る割合
- 画像の中に入った点に、`r62_seg.py` のラベル（最も近い RGB）を付け、b-1 の走査なら GT の姿勢で置いて、床・壁・天井のラベルの点の z の中央値・5%・95%
- 標本：b-1 の走査 272 本から等間隔に 20 本
- **向きの規則（回す前に固定）**：画像の中に入る割合が大きい方を使う。両方とも 5% 未満なら止める（S5）
build（§3-1・§3-2）
- b-2：`normalized/` に同じ時刻の姿勢がある走査。姿勢は N（その座標のまま）。b-1：S2 の GT に同じ時刻の行がある走査。姿勢は GT
- 各走査の点（有限のもの）に、orient で決めた向きで投影したラベルを付ける。画像の外・カメラの後ろの点は背景（0）
- 走査を時刻の順に並べ、最初の ceil(n·f) 本（f = 0.2・0.4・0.6・0.8・1.0）の点をつなぐ。
  `np.random.default_rng(0).choice(全点, 200000, replace=False)` で 200,000 点にする（`points_ply` は読み込みで標本化しないので、書き出す前に標本化した）
- 出力：`~/rizu/conslam/inputs/<b2|b1>_f<f>.ply`（色はクラスのパレット `labels.LABEL_COLORS`、法線なし）
refs（§3-3）
- 密な IFC（`conslam_ifc_dense5m_n.npz`）から、箱の xy（R57 報告 §4-2 の S1・S3 の小・大）と z ∈ [9.7, 14.1) で切り出す
- 天井：床のクラスで法線の z < −0.7 の点のうち、z ∈ [13.1, 14.1) は天井にし、z ∈ [9.7, 10.7) は外す（§2）
- 本研究の参照の読み込み（`io_utils.load_reference_cloud`、type ifc、`cache_path`）で読める npz にする（`spaces: []`、`storeys: []`、`inner_only: false`）
- Ω：`criterion_verdict.build_omega(cfg_b1_f1p0)`（`omega.npz` を作ったのと同じ関数）で、b-1 の f 1.0 から。b-2 用は A⁻¹ をかけた同じ点
- Ω_BIM：Ω（IFC の座標）のうち、参照の点から水平 0.3 m 以内の点（R39 と同じ規則）。参照ごと。Ω_小：小の箱の Ω_BIM（大にも使う）
check（§3-4、§4 S6）：b-2 の f 1.0 を A で、b-1 の f 1.0 をそのまま IFC の座標に置き、密な IFC への最近点の距離。クラスごと、参照の箱の中ごと。
  箱の xy の外にある点の割合（入力 × f × 参照）と r = 1 − 外の割合
configs（§3-5 の 1）：R49 の完全な方式の config（`r49_run.config("real", "m3_cor_a__E2", "full")`）を写し、source と reference だけを替える
出力：`~/rizu/conslam/out/r62_*.json`、`~/rizu/conslam/inputs/`、`~/rizu/conslam/refs/`、`Registration/configs/conslam/`
"""
import glob, json, math, os, sys, copy
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

REPO = os.path.expanduser("~/rizu/SNI-SLAM")
for p in ("Registration", "Registration/scripts", "Registration/scripts/r49"):
    sys.path.insert(0, os.path.join(REPO, p))
os.chdir(REPO)

D = "/mnt/d/rizu/SNI-SLAM/data/ConSLAM"
W = os.path.expanduser("~/rizu/conslam/work")
OUT = os.path.expanduser("~/rizu/conslam/out")
INP = os.path.expanduser("~/rizu/conslam/inputs")
REF = os.path.expanduser("~/rizu/conslam/refs")
IFC = D + "/m1743877/ConSLAM_Building_00_GroundTruth_V9_wall_segment.ifc"
FS = ["0p2", "0p4", "0p6", "0p8", "1p0"]
BOXES = {"S1_small": (136.5, 83.5, 146.5, 91.5), "S1_large": (136.5, 83.5, 156.5, 91.5),
         "S3_small": (177.5, 82.0, 187.5, 90.0), "S3_large": (167.5, 82.0, 187.5, 90.0)}
ZLO, ZHI = 9.7, 14.1
WH = (2064, 1544)


def read_pcd(p):
    with open(p, "rb") as f:
        while True:
            l = f.readline()
            if l.startswith(b"POINTS"):
                n = int(l.split()[1])
            if l.startswith(b"DATA"):
                break
        x = np.frombuffer(f.read(n * 12), dtype="<f4").reshape(n, 3).astype(np.float64)
    return x[np.isfinite(x).all(1)]


def calib():
    import yaml
    c = yaml.safe_load(open(D + "/Copy of data_calib/data_calib/calib_rgb.yaml"))
    K = np.array(c["camera_matrix"]["data"], dtype=np.float64).reshape(3, 3); dist = np.array(c["distortion_coefficients"]["data"], dtype=np.float64)
    txt = open(D + "/Copy of data_calib/data_calib/calib_lidar2rgb.txt").read()
    import re
    nums = [float(x) for x in re.findall(r"-?\d+\.\d+(?:e-?\d+)?|-?\d+\.\d+", txt.split("matrix:")[1].split("The above")[0])]
    M = np.array(nums[:16]).reshape(4, 4)
    return K, dist, M


def project(x_lidar, T_l2c, K, dist):
    import cv2
    xc = x_lidar @ T_l2c[:3, :3].T + T_l2c[:3, 3]
    front = xc[:, 2] > 0.05
    uv = np.full((len(xc), 2), -1.0)
    if front.any():
        u, _ = cv2.projectPoints(xc[front].reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, dist)
        uv[front] = u.reshape(-1, 2)
    inside = front & (uv[:, 0] >= 0) & (uv[:, 0] < WH[0]) & (uv[:, 1] >= 0) & (uv[:, 1] < WH[1])
    return uv, inside


def T_tum(row):
    T = np.eye(4); T[:3, :3] = Rotation.from_quat(row[4:8]).as_matrix(); T[:3, 3] = row[1:4]; return T


def scans():
    pcd = {int(os.path.basename(p)[:-4]): p for p in glob.glob(os.path.join(W, "lidar-pcd", "*.pcd"))}
    nz = {int(os.path.basename(p)[:-4]): p for p in glob.glob(os.path.join(W, "normalized", "*.txt"))}
    a = np.loadtxt(D + "/m1743877/S2_All_LiDAR_poses_v5_denser_T31.tum")
    gt = {int(round(r[0] * 1e6)): T_tum(r) for r in a}
    b2 = sorted((s, np.loadtxt(nz[s])) for s in pcd if s in nz)
    b1 = sorted((s, gt[s]) for s in pcd if s in gt)
    return pcd, b2, b1


def labels_for(stamp, x, T_l2c, K, dist, m, cache={}):
    import cv2
    r = m[str(stamp)]["rgb"]
    if r not in cache:
        cache.clear(); cache[r] = cv2.imread(os.path.join(W, "seg", "%d.png" % r), cv2.IMREAD_GRAYSCALE)
    img = cache[r]
    uv, inside = project(x, T_l2c, K, dist)
    lab = np.zeros(len(x), dtype=np.int64)
    ui = np.floor(uv[inside]).astype(int)
    lab[inside] = img[ui[:, 1], ui[:, 0]]
    return lab, inside


def orient():
    K, dist, M = calib(); pcd, b2, b1 = scans(); m = json.load(open(os.path.join(W, "scan_rgb.json")))["scans"]
    pick = [b1[int(round(i))] for i in np.linspace(0, len(b1) - 1, 20)]
    res = {"K": K.tolist(), "dist": dist.tolist(), "M": M.tolist()}
    for name, T in (("M", M), ("M_inv", np.linalg.inv(M))):
        n_all = n_in = 0; zs = {1: [], 3: [], 5: []}
        for s, Tw in pick:
            x = read_pcd(pcd[s]); lab, inside = labels_for(s, x, T, K, dist, m)
            n_all += len(x); n_in += int(inside.sum())
            y = x @ Tw[:3, :3].T + Tw[:3, 3]
            for c in zs:
                zs[c] += y[(lab == c) & inside, 2].tolist()
        res[name] = {"n_points": n_all, "frac_in_image": n_in / n_all,
                     "z_by_label": {k: ({"n": len(v), "median": float(np.median(v)), "p5": float(np.percentile(v, 5)), "p95": float(np.percentile(v, 95))} if v else None)
                                    for k, v in (("wall", zs[1]), ("floor", zs[3]), ("ceiling", zs[5]))}}
    res["chosen"] = "M" if res["M"]["frac_in_image"] >= res["M_inv"]["frac_in_image"] else "M_inv"
    res["stop_S5"] = max(res["M"]["frac_in_image"], res["M_inv"]["frac_in_image"]) < 0.05
    json.dump(res, open(os.path.join(OUT, "r62_orient.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ("K", "dist", "M")}, indent=1))


def build():
    from regbim.labels import LABEL_COLORS
    import open3d as o3d
    K, dist, M = calib(); o = json.load(open(os.path.join(OUT, "r62_orient.json")))
    T_l2c = M if o["chosen"] == "M" else np.linalg.inv(M)
    pcd, b2, b1 = scans(); m = json.load(open(os.path.join(W, "scan_rgb.json")))["scans"]
    os.makedirs(INP, exist_ok=True); res = {"orientation": o["chosen"]}
    for name, seq in (("b2", b2), ("b1", b1)):
        per = []
        for s, Tw in seq:
            x = read_pcd(pcd[s]); lab, inside = labels_for(s, x, T_l2c, K, dist, m)
            per.append((x @ Tw[:3, :3].T + Tw[:3, 3], lab, int(inside.sum())))
        res[name] = {"n_scans": len(seq), "t_first": seq[0][0], "t_last": seq[-1][0]}
        for f in FS:
            n = math.ceil(len(per) * float(f.replace("p", ".")))
            P = np.concatenate([p[0] for p in per[:n]]); L = np.concatenate([p[1] for p in per[:n]])
            idx = np.sort(np.random.default_rng(0).choice(len(P), min(200000, len(P)), replace=False))
            pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P[idx])); pc.colors = o3d.utility.Vector3dVector(LABEL_COLORS[L[idx]] / 255.0)
            path = os.path.join(INP, "%s_f%s.ply" % (name, f)); o3d.io.write_point_cloud(path, pc)
            cs = np.bincount(L[idx], minlength=6); ca = np.bincount(L, minlength=6)
            res[name]["f" + f] = {"n_scans": n, "n_points_concat": int(len(P)), "n_points_in_image_concat": int(sum(p[2] for p in per[:n])),
                                  "n_sampled": int(len(idx)), "class_counts_sampled": cs.tolist(), "class_frac_sampled": (cs / cs.sum()).tolist(),
                                  "labelled_frac_sampled": float(1 - cs[0] / cs.sum()), "class_counts_concat": ca.tolist(), "path": path}
            print(name, f, res[name]["f" + f]["n_scans"], res[name]["f" + f]["n_points_concat"], flush=True)
    json.dump(res, open(os.path.join(OUT, "r62_inputs.json"), "w"), indent=1, default=int)


def refs():
    import yaml
    sys.path.insert(0, os.path.join(REPO, "Registration", "scripts"))
    from regbim import io_utils
    from criterion_verdict import build_omega
    import r49_run as RR
    z = np.load(os.path.join(OUT, "conslam_ifc_dense5m_n.npz")); P = z["points"].astype(np.float64); L = z["labels"].copy(); N = z["normals"].astype(np.float64)
    down = (L == 3) & (N[:, 2] < -0.7)
    os.makedirs(REF, exist_ok=True); res = {"refs": {}}
    for name, b in BOXES.items():
        m = (P[:, 0] >= b[0]) & (P[:, 0] < b[2]) & (P[:, 1] >= b[1]) & (P[:, 1] < b[3]) & (P[:, 2] >= ZLO) & (P[:, 2] < ZHI)
        lab = L.copy(); lab[down & (P[:, 2] >= 13.1) & (P[:, 2] < 14.1)] = 5
        m &= ~(down & (P[:, 2] >= 9.7) & (P[:, 2] < 10.7))
        meta = {"source": "ifc", "path": IFC, "spaces_requested": [], "storeys_requested": [], "keep_classes": ["wall", "floor", "ceiling", "door", "window"],
                "r62_box": list(b), "r62_z": [ZLO, ZHI], "r62_note": "dense5m_n から切り出し、天井を付け直した（r62_build.refs）"}
        path = os.path.join(REF, "%s.npz" % name)
        np.savez_compressed(path, points=P[m], labels=lab[m], normals=N[m], meta=json.dumps(meta))
        res["refs"][name] = {"path": path, "n_points": int(m.sum()), "by_class": {int(k): int(v) for k, v in zip(*np.unique(lab[m], return_counts=True))}}
    json.dump(res, open(os.path.join(OUT, "r62_refs.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


def cfg_for(inp, f, ref):
    import r49_run as RR
    cfg = copy.deepcopy(RR.config("real", "m3_cor_a__E2", "full"))
    cfg["source"] = {"type": "points_ply", "mesh_path": os.path.join(INP, "%s_f%s.ply" % (inp, f))}
    cfg["reference"] = {"type": "ifc", "path": IFC, "cache_path": os.path.join(REF, "%s.npz" % ref), "spaces": [], "storeys": [],
                        "inner_only": False, "keep_classes": ["wall", "floor", "ceiling", "door", "window"], "n_points": 0, "seed": 0}
    return cfg


def omegas():
    from regbim import io_utils
    from criterion_verdict import build_omega
    A = np.array(json.load(open(os.path.join(OUT, "extra.json")))["b_normalized"]["A"])
    om_b1 = build_omega(cfg_for("b1", "1p0", "S1_small"), seed=0)
    om_b2 = om_b1 @ np.linalg.inv(A)[:3, :3].T + np.linalg.inv(A)[:3, 3]
    res = {"A": A.tolist(), "n_omega": int(len(om_b1)), "ref_load": {}, "omega_bim": {}}
    out = {"omega_b1": om_b1, "omega_b2": om_b2}
    for ref in BOXES:
        c = cfg_for("b1", "1p0", ref); dst = io_utils.load_reference_cloud(c)
        res["ref_load"][ref] = {"n_points": int(len(dst.points)), "by_class": {int(k): int(v) for k, v in zip(*np.unique(dst.labels, return_counts=True))},
                                "has_normals": dst.normals is not None}
        d = cKDTree(dst.points[:, :2]).query(om_b1[:, :2], k=1)[0]
        out["omega_bim_" + ref] = om_b1[d <= 0.3]; res["omega_bim"][ref] = int((d <= 0.3).sum())
    for s in ("S1", "S3"):
        out["omega_small_%s" % s] = out["omega_bim_%s_small" % s]; res["omega_bim"]["omega_small_%s" % s] = int(len(out["omega_small_%s" % s]))
    np.savez_compressed(os.path.join(OUT, "r62_omega.npz"), **out)
    json.dump(res, open(os.path.join(OUT, "r62_omega.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


def check():
    from regbim import io_utils
    A = np.array(json.load(open(os.path.join(OUT, "extra.json")))["b_normalized"]["A"])
    z = np.load(os.path.join(OUT, "conslam_ifc_dense5m_n.npz")); t = cKDTree(z["points"].astype(np.float64))
    def q(d):
        return {"n": int(len(d)), "median": float(np.median(d)), "p90": float(np.percentile(d, 90)), "p95": float(np.percentile(d, 95))} if len(d) else None
    res = {"dist": {}, "outside": {}}
    for inp in ("b2", "b1"):
        for f in FS:
            c = io_utils.load_source_cloud(cfg_for(inp, f, "S1_small"))
            X = c.points @ A[:3, :3].T + A[:3, 3] if inp == "b2" else c.points
            if f == "1p0":
                d = t.query(X, k=1)[0]
                e = {"all": q(d), "by_class": {int(k): q(d[c.labels == k]) for k in np.unique(c.labels)}}
                for ref, b in BOXES.items():
                    mm = (X[:, 0] >= b[0]) & (X[:, 0] < b[2]) & (X[:, 1] >= b[1]) & (X[:, 1] < b[3])
                    e[ref] = q(d[mm])
                res["dist"][inp] = e
            for ref, b in BOXES.items():
                mm = (X[:, 0] >= b[0]) & (X[:, 0] < b[2]) & (X[:, 1] >= b[1]) & (X[:, 1] < b[3])
                res["outside"]["%s|%s|%s" % (inp, f, ref)] = {"outside_frac": float(1 - mm.mean()), "r": float(mm.mean())}
    res["stop_S6"] = any(res["dist"][i]["all"]["median"] > 0.3 for i in res["dist"])
    json.dump(res, open(os.path.join(OUT, "r62_check.json"), "w"), indent=1)
    print(json.dumps(res["dist"], indent=1)[:3000]); print("stop", res["stop_S6"])


def configs():
    import yaml
    d = "Registration/configs/conslam"; os.makedirs(d, exist_ok=True); n = 0
    for inp in ("b2", "b1"):
        for f in FS:
            for ref in BOXES:
                yaml.safe_dump(cfg_for(inp, f, ref), open(os.path.join(d, "%s_f%s__%s.yaml" % (inp, f, ref)), "w"), sort_keys=False, allow_unicode=True)
                n += 1
    print("configs", n)


if __name__ == "__main__":
    {"orient": orient, "build": build, "refs": refs, "omegas": omegas, "check": check, "configs": configs}[sys.argv[1]]()
