"""R65 §2：ConSLAM の目視用の PLY の書き出しと、縮尺の確認（sni-slam の環境、ブランチ `exp/r65`）。**回す前に commit する。位置合わせは回さない。**

使い方：python r65_view.py ply → scale

ply（§2 の表の 1〜10）：出力 `~/rizu/conslam/view/`。PLY は binary little endian、x・y・z は float32、色は uchar の RGB
- 1 `ifc_dense.ply`：`out/conslam_ifc_dense5m_n.npz`（R57 の 5,000,000 点）のうち z ∈ [9.7, 14.1)。色は `labels.LABEL_COLORS`（ラベルは npz のまま。柱などは背景＝灰）
- 2 `ref_<参照>.ply`：`refs_fix/<参照>.npz`（R63 §8 で直した参照）。色はクラス
- 3 `ref_boxes.ply`：R62 の 4 つの箱（xy は `r62_build.BOXES`、z ∈ [9.7, 14.1)）の 12 本の辺を 0.05 m おきの点で。小は赤 (255, 0, 0)、大は橙 (230, 159, 0)
- 4 `input_b2_f1p0_normalized.ply`：`inputs_fix/b2_f1p0.ply` の点とクラスの色（`normalized/` の座標のまま）
- 5 `input_b2_f1p0_onIFC.ply`：4 の点に正解 A（`out/extra.json` の `b_normalized.A`）をかけたもの。オレンジ (230, 159, 0)
- 6 `input_b2_f1p0_onIFC_labels.ply`：5 と同じ点をクラスの色で
- 7 `tls_sample.ply`：`r57_coords.read_tls()`（R57 と同じ seed 0 の 2,000,000 点）。灰 (128, 128, 128)
- 8 `gt_traj_S2.ply`：`m1743877/S2_All_LiDAR_poses_v5_denser_T31.tum` の 317 行の位置を、隣どうしを結ぶ線として 0.05 m おきの点で。黒 (0, 0, 0)
- 9 `fail_condB_<参照>_k0.ply`：`r63_data/results_fix/full/b2_f1p0__<参照>.json` の k 0 の `T_est`・`P` で、4 の点を T_est · P で置いたもの。赤 (213, 94, 0)
- 10 `condA_<参照>_k<k>.ply`：`r63_data/resultsA_fix/full/b2_f1p0__<参照>.json` の `T_est`・`P`（P_eff）で 4 の点を置いたもの。青 (0, 114, 178)。
  S3_small は k 1（基準内）。S1_small・S1_large は、e_R < 45° の試行のうち k が最小のもの（規則はいま固定）
- 記録：各ファイルの点の数・大きさ、9・10 の各試行の e_Ω・e_R（`per_trial(T_est, P, A, Ω_b2)`、Ω は `r63fix_omega.npz`）。`~/rizu/conslam/out/r65_view.json`
scale（§2-1）：
- 1：R57 と同じ 204 組（S2 の GT と `normalized/` の時刻が一致する行の位置）に、剛体の Umeyama（今の A、`r57_extra.umeyama_rigid`）と、
  縮尺を含む Umeyama（相似変換）を当てる。縮尺と残差 |GT の位置 − T · N の位置|（中央値・95%・最大）
- 2：柱（`out/elements.json` の IfcColumn の外接箱）のうち、z の範囲が [10.7, 13.1) を含み、入力 5 の点が「外接箱を水平に 0.15 m 広げた範囲 × z ∈ [10.7, 13.1)」に
  50 点以上ある柱を使う。IFC の柱の中心 = 外接箱の xy の中点。入力の塊の中心 = その範囲の入力 5 の点の xy の外接矩形の中点。
  x 方向の組：IFC の中心の y の差が 0.3 m 以内で x の差が 3 m 以上の、x の順に隣り合う柱の組。y 方向も同じ。距離の比 = 入力の中心どうしの距離 / IFC の中心どうしの距離
  （規則はいま固定。塊の外接矩形の大きさも書く）
- 3：IFC の長さの単位：`ifc_export.length_scale_to_m`（`Registration/scripts/ifc_export.py:93-`）を ConSLAM の IFC に当てた値（`out/ifc_info.json` の `unit`）と、
  幾何（`ifcopenshell.geom`）が m を返すこと（`out/elements.json` の `unit_scale_applied_by_geom`）
出力：`~/rizu/conslam/out/r65_scale.json`
"""
import glob, json, os, sys
import numpy as np

REPO = os.path.expanduser("~/rizu/SNI-SLAM")
for p in ("Registration", "Registration/scripts", "Registration/scripts/r57", "Registration/scripts/r62"):
    sys.path.insert(0, os.path.join(REPO, p))
os.chdir(REPO)
from regbim.labels import LABEL_COLORS, color_to_label         # noqa: E402

CO = os.path.expanduser("~/rizu/conslam"); OUT = CO + "/out"; VIEW = CO + "/view"
D = "/mnt/d/rizu/SNI-SLAM/data/ConSLAM"
REFS = ["S1_small", "S1_large", "S3_small", "S3_large"]
BOXES = {"S1_small": (136.5, 83.5, 146.5, 91.5), "S1_large": (136.5, 83.5, 156.5, 91.5),
         "S3_small": (177.5, 82.0, 187.5, 90.0), "S3_large": (167.5, 82.0, 187.5, 90.0)}
A = np.array(json.load(open(OUT + "/extra.json", encoding="utf-8"))["b_normalized"]["A"])


def write_ply(path, xyz, rgb):
    xyz = np.asarray(xyz, dtype="<f4"); rgb = np.asarray(rgb)
    if rgb.ndim == 1:
        rgb = np.tile(rgb, (len(xyz), 1))
    rec = np.empty(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
    rec["x"], rec["y"], rec["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    rec["r"], rec["g"], rec["b"] = rgb[:, 0].astype(np.uint8), rgb[:, 1].astype(np.uint8), rgb[:, 2].astype(np.uint8)
    head = ("ply\nformat binary_little_endian 1.0\nelement vertex %d\nproperty float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n" % len(xyz)).encode()
    with open(path, "wb") as f:
        f.write(head); f.write(rec.tobytes())
    return {"file": os.path.basename(path), "n_points": int(len(xyz)), "bytes": os.path.getsize(path)}


def app(T, X):
    return X @ T[:3, :3].T + T[:3, 3]


def polyline(P, step=0.05):
    out = []
    for a, b in zip(P[:-1], P[1:]):
        n = max(int(np.ceil(np.linalg.norm(b - a) / step)), 1)
        out.append(a + (b - a) * np.linspace(0, 1, n, endpoint=False)[:, None])
    out.append(P[-1:]); return np.concatenate(out)


def box_edges(b, z0=9.7, z1=14.1):
    x0, y0, x1, y1 = b; c = np.array([[x, y, z] for z in (z0, z1) for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))])
    E = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    return np.concatenate([polyline(c[[i, j]]) for i, j in E])


def load_input():
    import open3d as o3d
    pc = o3d.io.read_point_cloud(CO + "/inputs_fix/b2_f1p0.ply")
    return np.asarray(pc.points), color_to_label(np.asarray(pc.colors))


def ply():
    sys.path.insert(0, os.path.join(REPO, "Registration/scripts"))
    from r41_rescore import per_trial
    from r57_coords import read_tls
    os.makedirs(VIEW, exist_ok=True); res = {"files": [], "trials": {}}
    z = np.load(OUT + "/conslam_ifc_dense5m_n.npz"); P = z["points"].astype(np.float64); L = z["labels"]
    m = (P[:, 2] >= 9.7) & (P[:, 2] < 14.1)
    res["files"].append(write_ply(VIEW + "/ifc_dense.ply", P[m], LABEL_COLORS[L[m]]))
    for r in REFS:
        q = np.load(CO + "/refs_fix/%s.npz" % r)
        res["files"].append(write_ply(VIEW + "/ref_%s.ply" % r, q["points"], LABEL_COLORS[q["labels"]]))
    bx = [(box_edges(b), (255, 0, 0) if "small" in r else (230, 159, 0)) for r, b in BOXES.items()]
    res["files"].append(write_ply(VIEW + "/ref_boxes.ply", np.concatenate([e for e, _ in bx]), np.concatenate([np.tile(c, (len(e), 1)) for e, c in bx])))
    X, lab = load_input()
    res["files"].append(write_ply(VIEW + "/input_b2_f1p0_normalized.ply", X, LABEL_COLORS[lab]))
    Y = app(A, X)
    res["files"].append(write_ply(VIEW + "/input_b2_f1p0_onIFC.ply", Y, (230, 159, 0)))
    res["files"].append(write_ply(VIEW + "/input_b2_f1p0_onIFC_labels.ply", Y, LABEL_COLORS[lab]))
    tls, info = read_tls()
    res["files"].append(write_ply(VIEW + "/tls_sample.ply", tls, (128, 128, 128))); res["tls_info"] = info
    a = np.loadtxt(D + "/m1743877/S2_All_LiDAR_poses_v5_denser_T31.tum")
    res["files"].append(write_ply(VIEW + "/gt_traj_S2.ply", polyline(a[:, 1:4]), (0, 0, 0))); res["gt_rows"] = int(len(a))
    OM = np.load(OUT + "/r63fix_omega.npz")["omega_b2"]
    def place(path, k, name, color):
        r = json.load(open(path))["rows"][str(k)]; T = np.asarray(r["T_est"]); Pk = np.asarray(r["P"])
        e = per_trial(T, Pk, A, OM, None)
        res["files"].append(write_ply(VIEW + "/" + name, app(T @ Pk, X), color))
        res["trials"][name] = {"record": path, "k": k, "eO": float(e["eO"]), "eR": float(e["eR"]), "run_eO": r["run_eO"]}
    RB = os.path.expanduser("~/rizu/r63_data/results_fix/full/b2_f1p0__%s.json"); RA = os.path.expanduser("~/rizu/r63_data/resultsA_fix/full/b2_f1p0__%s.json")
    for r in REFS:
        place(RB % r, 0, "fail_condB_%s_k0.ply" % r, (213, 94, 0))
    place(RA % "S3_small", 1, "condA_S3_small_k1.ply", (0, 114, 178))
    for r in ("S1_small", "S1_large"):
        rows = json.load(open(RA % r))["rows"]
        k = min(int(i) for i, v in rows.items() if v["run_eR"] < 45)
        place(RA % r, k, "condA_%s_k%d.ply" % (r, k), (0, 114, 178))
    res["total_bytes"] = sum(f["bytes"] for f in res["files"])
    json.dump(res, open(OUT + "/r65_view.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(res, indent=1, ensure_ascii=False))


def umeyama_sim(a, b):
    ma, mb = a.mean(0), b.mean(0); A0, B0 = a - ma, b - mb
    H = A0.T @ B0 / len(a); U, S, Vt = np.linalg.svd(H); Dg = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ Dg @ U.T; s = np.trace(np.diag(S) @ Dg) / (A0 ** 2).sum(1).mean()
    T = np.eye(4); T[:3, :3] = s * R; T[:3, 3] = mb - s * R @ ma; return T, float(s)


def scale():
    from r57_extra import umeyama_rigid
    res = {}
    a = np.loadtxt(D + "/m1743877/S2_All_LiDAR_poses_v5_denser_T31.tum")
    nz = {int(os.path.splitext(os.path.basename(p))[0]): np.loadtxt(p) for p in glob.glob(CO + "/work/normalized/*.txt")}
    pairs = [(i, nz[int(round(a[i, 0] * 1e6))]) for i in range(len(a)) if int(round(a[i, 0] * 1e6)) in nz]
    src = np.array([N[:3, 3] for _, N in pairs]); dst = np.array([a[i, 1:4] for i, _ in pairs])
    Tr = umeyama_rigid(src, dst); Ts, s = umeyama_sim(src, dst)
    def rs(T):
        d = np.linalg.norm(app(T, src) - dst, axis=1); return {"median": float(np.median(d)), "p95": float(np.percentile(d, 95)), "max": float(d.max())}
    res["1"] = {"n_pairs": len(pairs), "rigid": {"resid": rs(Tr), "same_as_extra_A": float(np.abs(Tr - A).max())},
                "similarity": {"scale": s, "resid": rs(Ts)}, "src_extent_m": float(np.linalg.norm(src.max(0) - src.min(0)))}
    # 2
    el = json.load(open(OUT + "/elements.json", encoding="utf-8"))["elements"]
    cols = [c for c in el if c["type"] == "IfcColumn" and c["min"][2] <= 10.7 and c["max"][2] >= 13.1]
    X, _ = load_input(); Y = app(A, X); Yb = Y[(Y[:, 2] >= 10.7) & (Y[:, 2] < 13.1)]
    used = []
    for c in cols:
        lo, hi = np.array(c["min"][:2]) - 0.15, np.array(c["max"][:2]) + 0.15
        mm = (Yb[:, 0] >= lo[0]) & (Yb[:, 0] <= hi[0]) & (Yb[:, 1] >= lo[1]) & (Yb[:, 1] <= hi[1])
        if mm.sum() >= 50:
            pts = Yb[mm][:, :2]
            used.append({"id": c["id"], "ifc_center": ((np.array(c["min"][:2]) + np.array(c["max"][:2])) / 2).tolist(),
                         "ifc_size": (np.array(c["max"][:2]) - np.array(c["min"][:2])).tolist(), "n_input": int(mm.sum()),
                         "input_center": ((pts.min(0) + pts.max(0)) / 2).tolist(), "input_bbox_size": (pts.max(0) - pts.min(0)).tolist()})
    res["2"] = {"n_columns_in_band": len(cols), "n_used": len(used), "columns": used, "pairs": {"x": [], "y": []}}
    for ax, other, key in ((0, 1, "x"), (1, 0, "y")):
        for i, ci in enumerate(used):
            row = sorted([cj for cj in used if abs(cj["ifc_center"][other] - ci["ifc_center"][other]) <= 0.3], key=lambda c: c["ifc_center"][ax])
            for c1, c2 in zip(row[:-1], row[1:]):
                if c1 is ci:
                    dI = np.linalg.norm(np.subtract(c2["ifc_center"], c1["ifc_center"])); dP = np.linalg.norm(np.subtract(c2["input_center"], c1["input_center"]))
                    if abs(c2["ifc_center"][ax] - c1["ifc_center"][ax]) >= 3.0:
                        res["2"]["pairs"][key].append({"ids": [c1["id"], c2["id"]], "ifc_dist": float(dI), "input_dist": float(dP), "ratio": float(dP / dI)})
    res["3"] = {"ifc_info_unit": json.load(open(OUT + "/ifc_info.json", encoding="utf-8"))["unit"],
                "elements_unit_scale_applied_by_geom": json.load(open(OUT + "/elements.json", encoding="utf-8"))["unit_scale_applied_by_geom"]}
    json.dump(res, open(OUT + "/r65_scale.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in res.items() if k != "2"}, indent=1, ensure_ascii=False))
    print("cols", res["2"]["n_columns_in_band"], res["2"]["n_used"]); print(json.dumps(res["2"]["pairs"], indent=1))


if __name__ == "__main__":
    {"ply": ply, "scale": scale}[sys.argv[1]]()
