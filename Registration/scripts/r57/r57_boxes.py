"""R57 §3-5（§7-2 の指示）：範囲（箱）で切る部分 BIM の候補と、天井のクラスの確認（sni-slam の環境）。**回す前に commit する。規則は回す前に固定した。**

この階の高さ：床の付近 z ∈ [9.7, 10.7)、壁の高さ [10.7, 13.1)、天井の付近 [13.1, 14.1)（`r57_extra.py` の d と同じ帯）
小の箱（規則）
- 探す範囲：S2 の GT の軌跡の xy の範囲を 5 m ずつ広げた矩形。箱は IFC の軸にそろえ、(幅, 奥行) ∈ {(6,8),(8,6),(8,8),(7,10),(10,7),(8,10),(10,8)} m（面積 48〜80 m²）、
  左下の角は 0.5 m おき
- 辺が壁で囲まれているか：密な IFC の壁のクラスの点のうち z が壁の高さの帯にある点を、xy で KD 木にする。各辺に 0.25 m おきに標本点を置き、
  水平距離 0.3 m 以内に壁の点がある標本点の割合が 0.5 以上なら、その辺は「囲まれている」
- 囲まれた辺が 3 以上の箱のうち、GT の軌跡の行（317 行、ほぼ 1 s おき）が箱の xy の中にある割合（時間の割合）が大きい順に、
  すでに選んだ箱と重ならないものを 3 つ。同じ値なら、囲まれた辺の数が多い方、次に面積が大きい方、次に x・y が小さい方
大の箱：小の箱の 4 方向の隣（同じ大きさで辺を共有）のうち、囲まれた辺（大の箱の外周で数える）が多いもの、同じなら時間の割合が大きいものを足した矩形
各組の量：床の面積（床のクラスの点で z が床の付近の帯にあるものを 0.1 m 格子に落とし、箱の中で占めたマスの数 × 0.01 m²）、
要素の数（`elements.json` の箱の中心の xy が箱の中にあり、中心の z が [9.7, 14.1) の要素。種類ごと）、
系列 2 の LiDAR の点のうち箱の xy の外にある割合（GT の行と同じ時刻の走査 272 本を GT の姿勢で置いた全点）、GT の軌跡が箱を通る時間の割合
天井：`conslam_ifc_all.npz`（参照の作り方の 20 万点）で、床のクラスの点のうち法線の z 成分が −0.7 より小さい点（スラブの下面）を天井とした場合の、クラスごとの数
出力：`~/rizu/conslam/out/boxes.json`、図 `boxes.png`
"""
import glob, json, os
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

OUT = os.path.expanduser("~/rizu/conslam/out")
W = os.path.expanduser("~/rizu/conslam/work")
D = "/mnt/d/rizu/SNI-SLAM/data/ConSLAM"
SIZES = [(6, 8), (8, 6), (8, 8), (7, 10), (10, 7), (8, 10), (10, 8)]
FLOOR_Z, WALL_Z = (9.7, 10.7), (10.7, 13.1)


def read_pcd(p):
    with open(p, "rb") as f:
        while True:
            l = f.readline()
            if l.startswith(b"POINTS"):
                n = int(l.split()[1])
            if l.startswith(b"DATA"):
                break
        return np.frombuffer(f.read(n * 12), dtype="<f4").reshape(n, 3).astype(np.float64)


def sides_covered(box, tree):
    x0, y0, x1, y1 = box; n = 0
    for (ax, ay, bx, by) in ((x0, y0, x1, y0), (x1, y0, x1, y1), (x1, y1, x0, y1), (x0, y1, x0, y0)):
        L = np.hypot(bx - ax, by - ay); m = max(2, int(L / 0.25) + 1)
        s = np.stack([np.linspace(ax, bx, m), np.linspace(ay, by, m)], 1)
        d = tree.query(s, k=1)[0]
        n += int((d <= 0.3).mean() >= 0.5)
    return n


def inside(xy, box):
    return (xy[:, 0] >= box[0]) & (xy[:, 0] < box[2]) & (xy[:, 1] >= box[1]) & (xy[:, 1] < box[3])


def overlap(a, b):
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def main():
    z = np.load(os.path.join(OUT, "conslam_ifc_dense5m.npz")); P = z["points"].astype(np.float64); L = z["labels"]
    wall = P[(L == 1) & (P[:, 2] >= WALL_Z[0]) & (P[:, 2] < WALL_Z[1])]; wt = cKDTree(wall[:, :2])
    fl = P[(L == 3) & (P[:, 2] >= FLOOR_Z[0]) & (P[:, 2] < FLOOR_Z[1])]
    a = np.loadtxt(os.path.join(D, "m1743877", "S2_All_LiDAR_poses_v5_denser_T31.tum")); traj = a[:, 1:3]
    lo, hi = traj.min(0) - 5, traj.max(0) + 5
    cands = []
    for w, h in SIZES:
        for x0 in np.arange(np.floor(lo[0] * 2) / 2, hi[0] - w + 1e-9, 0.5):
            for y0 in np.arange(np.floor(lo[1] * 2) / 2, hi[1] - h + 1e-9, 0.5):
                box = (float(x0), float(y0), float(x0 + w), float(y0 + h))
                tf = float(inside(traj, box).mean())
                if tf == 0:
                    continue
                ns = sides_covered(box, wt)
                if ns >= 3:
                    cands.append((tf, ns, w * h, -x0, -y0, box))
    cands.sort(key=lambda c: (-c[0], -c[1], -c[2], -c[3], -c[4]))
    small = []
    for c in cands:
        if all(not overlap(c[5], s[5]) for s in small):
            small.append(c)
        if len(small) == 3:
            break
    # LiDAR（GT で置いた 272 本）
    pcd = {int(os.path.splitext(os.path.basename(p))[0]): p for p in glob.glob(os.path.join(W, "lidar-pcd", "*.pcd"))}
    pts = []
    for row in a:
        k = int(round(row[0] * 1e6))
        if k in pcd:
            x = read_pcd(pcd[k]); x = x[np.isfinite(x).all(1)]
            pts.append(x @ Rotation.from_quat(row[4:8]).as_matrix().T + row[1:4])
    pts = np.concatenate(pts)[:, :2]
    els = json.load(open(os.path.join(OUT, "elements.json")))["elements"]
    ec = np.array([(np.array(e["min"]) + np.array(e["max"])) / 2 for e in els]); et = np.array([e["type"] for e in els])
    occ = set(map(tuple, np.floor(fl[:, :2] / 0.1).astype(int)))
    def metrics(box):
        cells = [c for c in occ if box[0] <= (c[0] + 0.5) * 0.1 < box[2] and box[1] <= (c[1] + 0.5) * 0.1 < box[3]]
        m = inside(ec[:, :2], box) & (ec[:, 2] >= 9.7) & (ec[:, 2] < 14.1)
        from collections import Counter
        return {"box": box, "box_area_m2": (box[2] - box[0]) * (box[3] - box[1]), "floor_area_m2": len(cells) * 0.01,
                "n_elements": dict(Counter(et[m].tolist())), "sides_covered": sides_covered(box, wt),
                "lidar_outside_frac": float(1 - inside(pts, box).mean()), "gt_time_frac": float(inside(traj, box).mean())}
    res = {"n_candidates_small": len(cands), "n_lidar_points": int(len(pts)), "pairs": []}
    for c in small:
        b = c[5]; w, h = b[2] - b[0], b[3] - b[1]
        nb = [(b[0] + w, b[1], b[2] + w, b[3]), (b[0] - w, b[1], b[2] - w, b[3]), (b[0], b[1] + h, b[2], b[3] + h), (b[0], b[1] - h, b[2], b[3] - h)]
        big = [(min(b[0], n[0]), min(b[1], n[1]), max(b[2], n[2]), max(b[3], n[3])) for n in nb]
        big.sort(key=lambda B: (-sides_covered(B, wt), -float(inside(traj, B).mean())))
        res["pairs"].append({"small": metrics(b), "large": metrics(big[0])})
    # 天井
    r = np.load(os.path.join(OUT, "conslam_ifc_all.npz")); lab = r["labels"].copy(); nz = r["normals"][:, 2]
    lab2 = lab.copy(); lab2[(lab == 3) & (nz < -0.7)] = 5
    names = {1: "wall", 2: "door", 3: "floor", 4: "window", 5: "ceiling"}
    zz = r["points"][:, 2]
    res["ceiling_relabel"] = {"all": {names[int(k)]: int((lab2 == k).sum()) for k in np.unique(lab2)},
                              "z_in_[9.7,14.1)": {names[int(k)]: int(((lab2 == k) & (zz >= 9.7) & (zz < 14.1)).sum()) for k in np.unique(lab2)}}
    json.dump(res, open(os.path.join(OUT, "boxes.json"), "w"), indent=1)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig, ax = plt.subplots(figsize=(14, 7))
    rng = np.random.default_rng(0)
    ww = wall[rng.choice(len(wall), min(len(wall), 200000), replace=False)]
    ax.scatter(ww[:, 0], ww[:, 1], s=0.2, c="#0072B2", label="IFC wall (level band)")
    ax.plot(traj[:, 0], traj[:, 1], "k-", lw=0.8, label="S2 GT")
    for i, p in enumerate(res["pairs"]):
        for kind, col in (("large", "#E69F00"), ("small", "#D55E00")):
            b = p[kind]["box"]; ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec=col, lw=2))
        b = p["small"]["box"]; ax.text(b[0], b[3] + 0.3, "S%d" % (i + 1), color="#D55E00")
    ax.set_xlim(lo[0] - 5, hi[0] + 5); ax.set_ylim(lo[1] - 5, hi[1] + 5); ax.set_aspect("equal"); ax.legend(markerscale=10)
    ax.set_title("R57 box candidates (red: small, orange: large)"); fig.tight_layout(); fig.savefig(os.path.join(OUT, "boxes.png"), dpi=110)
    print(json.dumps(res, indent=1)[:4000])


if __name__ == "__main__":
    main()
