"""R48 段階0：部屋分けの規則・部分参照・STSL 用の BIM 入力・平面図（libim の環境）。**登録はしない。**

規則（固定。手で選ばない）
1. 床の高さ：floor の点の z を 0.05 m の刻みで数え、最も多い刻みの階を主の階とする（apartment_0 は 2 階建て。主の階だけを使う）
2. 0.1 m の格子：床の格子 = 主の階の floor の点がある格子（1 格子分ふさぎ、内側の穴＝家具の下で床の点が無い所を埋める）。壁の格子 = wall と door の点のうち床の上 0.3〜1.8 m の点がある格子を（扉も部屋の境目にする）
   1 格子分ふくらませたもの。通れる領域 = 床の格子 − 壁の格子
3. 部屋：通れる領域の距離変換（壁・外からの距離）で 0.5 m 以上の格子の連結成分を種にし（幅 1 m 未満の出入口で切れる）、
   watershed（−距離、通れる領域の中）で広げる。2 m² 未満の部屋は捨てる
4. 部屋のつながり：2 つの部屋の格子を 0.4 m ふくらませて重なれば隣
5. 部分参照（大きさの段 n = 1, 2, 3, …）：n = 1 は最も広い部屋。n+1 は、選んだ部屋のどれかに隣る部屋のうち最も広いものを足す。
   最後の段は全部の部屋。参照の点 = 選んだ部屋の格子を 0.3 m ふくらませた範囲に xy が入る構造 5 クラスの点
6. STSL 用の BIM 入力：参照の壁の点から 2D 線分（`lines_from_walls_v2`）。z は主の階の床を 0 にする
   `lines_from_walls_v2`（R46 の `lines_from_walls` の直し）：法線の向き（1° 刻み）と法線方向の位置（0.05 m 刻み）の刻みのうち、
   点の数がしきい値（向き：水平な壁の点の 0.5% と 50 の大きい方、位置：50）以上のものだけを、隣どうしまとめる。
   R46 の版はしきい値が無く、法線がばらつく Replica では全部の向きが 1 つにまとまった（IFC は法線がばらつかないので影響しない）
出力：`~/rizu/r48_data/rooms/<scene>.json`、`~/rizu/r48_data/BIM/<scene>_n<段>/`、図 `~/rizu/r48_data/fig/<scene>.png`
使い方：python r48_rooms.py [scene ...]
"""
import json, os, sys
import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import watershed

D = os.path.expanduser("~/rizu/r48_data")
SCENES = ["apartment_0", "apartment_1", "apartment_2", "frl_apartment_0", "frl_apartment_1", "frl_apartment_2",
          "frl_apartment_3", "frl_apartment_4", "frl_apartment_5", "hotel_0"]
WALL, DOOR, FLOOR, WINDOW, CEIL = 1, 2, 3, 4, 5
CELL = 0.1


def merge_runs(idx, n_wrap=None):
    """整数の刻みの集合を、隣どうし（n_wrap があれば両端も）でまとめる"""
    idx = sorted(set(int(i) for i in idx)); groups, cur = [], []
    for i in idx:
        if cur and i - cur[-1] > 1:
            groups.append(cur); cur = []
        cur.append(i)
    if cur:
        groups.append(cur)
    if n_wrap and len(groups) > 1 and groups[0][0] == 0 and groups[-1][-1] == n_wrap - 1:
        groups[0] = groups[-1] + groups[0]; groups.pop()
    return groups


def lines_from_walls_v2(P, N, ang_bin=1.0, off_bin=0.05, frac=0.005, min_pts=50, gap=0.3, min_len=0.3):
    h = np.abs(N[:, 2]) < 0.3; P, N = P[h], N[h]
    if len(P) == 0:
        return np.zeros((0, 4))
    ang = np.degrees(np.arctan2(N[:, 1], N[:, 0])) % 180.0
    nb = int(round(180 / ang_bin)); ab = np.minimum((ang / ang_bin).astype(int), nb - 1)
    cnt = np.bincount(ab, minlength=nb); thr = max(min_pts, frac * len(P))
    segs = []
    for grp in merge_runs(np.flatnonzero(cnt >= thr), n_wrap=nb):
        m = np.isin(ab, grp)
        a = ang[m]; ref = a[0]; a = (a - ref + 90) % 180 - 90 + ref           # 0/180 の継ぎ目をまたぐ群のため
        th = np.radians(np.median(a)); n = np.array([np.cos(th), np.sin(th)]); t = np.array([-n[1], n[0]])
        Q = P[m][:, :2]; off = Q @ n; ob = np.floor(off / off_bin).astype(int)
        oc = {}
        for b in ob:
            oc[b] = oc.get(b, 0) + 1
        for og in merge_runs([b for b, c in oc.items() if c >= min_pts]):
            mo = np.isin(ob, og)
            d = float(np.median(off[mo])); s = np.sort(Q[mo] @ t)
            cuts = np.flatnonzero(np.diff(s) > gap)
            for lo, hi in zip(np.r_[0, cuts + 1], np.r_[cuts, len(s) - 1]):
                if s[hi] - s[lo] >= min_len:
                    pa, pb = d * n + s[lo] * t, d * n + s[hi] * t
                    segs.append([pa[0], pa[1], pb[0], pb[1]])
    return np.asarray(segs).reshape(-1, 4)


def write_pcd(path, pts):
    pts = np.asarray(pts, dtype=np.float32)
    with open(path, "w") as f:
        f.write("# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n")
        f.write("WIDTH %d\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS %d\nDATA ascii\n" % (len(pts), len(pts)))
        np.savetxt(f, pts, fmt="%.4f")


def segment(scene):
    z = np.load(os.path.join(D, "ref", scene + ".npz")); P, L, N = z["points"].astype(np.float64), z["labels"], z["normals"].astype(np.float64)
    fz = P[L == FLOOR, 2]; hb = np.floor(fz / 0.05).astype(int); zb = np.bincount(hb - hb.min()).argmax() + hb.min()
    z0 = float(np.median(fz[np.abs(hb - zb) <= 2]))
    lvl = (P[:, 2] > z0 - 0.3) & (P[:, 2] < z0 + 3.0)                          # 主の階
    xy0 = P[lvl, :2].min(0) - 1.0; shp = np.ceil((P[lvl, :2].max(0) + 1.0 - xy0) / CELL).astype(int) + 1
    ij = lambda Q: np.floor((Q[:, :2] - xy0) / CELL).astype(int)
    floor = np.zeros(shp, bool); fi = ij(P[(L == FLOOR) & (np.abs(P[:, 2] - z0) < 0.15)]); floor[fi[:, 0], fi[:, 1]] = True
    floor = ndi.binary_fill_holes(ndi.binary_closing(floor, iterations=1))
    wall = np.zeros(shp, bool); wm = np.isin(L, [WALL, DOOR]) & (P[:, 2] > z0 + 0.3) & (P[:, 2] < z0 + 1.8); wi = ij(P[wm]); wall[wi[:, 0], wi[:, 1]] = True
    wall = ndi.binary_dilation(wall, iterations=1)
    free = floor & ~wall
    dist = ndi.distance_transform_edt(free) * CELL
    seeds, ns = ndi.label(dist >= 0.5)
    rooms = watershed(-dist, seeds, mask=free)
    areas = {r: float((rooms == r).sum() * CELL * CELL) for r in range(1, ns + 1)}
    keep = [r for r, a in sorted(areas.items(), key=lambda x: -x[1]) if a >= 2.0]
    dil = {r: ndi.binary_dilation(rooms == r, iterations=4) for r in keep}
    adj = {r: [q for q in keep if q != r and (dil[r] & (rooms == q)).any()] for r in keep}
    return dict(P=P, L=L, N=N, z0=z0, xy0=xy0, shp=shp, rooms=rooms, keep=keep, areas=areas, adj=adj, floor=floor, wall=wall, ij=ij)


def main():
    scenes = sys.argv[1:] or SCENES
    os.makedirs(os.path.join(D, "rooms"), exist_ok=True); os.makedirs(os.path.join(D, "fig"), exist_ok=True)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm
    for f in ("/mnt/c/Windows/Fonts/arial.ttf", "/home/student/plotconda/meiryo_ui.ttf"):
        fm.fontManager.addfont(f)
    plt.rcParams["font.family"] = ["Arial", "Meiryo UI"]; plt.rcParams["font.size"] = 8
    for scene in scenes:
        g = segment(scene); P, L, N, z0 = g["P"], g["L"], g["N"], g["z0"]
        order = [g["keep"][0]] if g["keep"] else []
        steps = [list(order)]
        while len(order) < len(g["keep"]):
            cand = sorted({q for r in order for q in g["adj"][r]} - set(order), key=lambda q: -g["areas"][q])
            if not cand:
                break
            order.append(cand[0]); steps.append(list(order))
        if steps and len(steps[-1]) < len(g["keep"]):
            steps.append(list(g["keep"]))                                   # つながらない部屋が残れば、最後に全部
        cell = g["ij"](P); inb = (cell[:, 0] >= 0) & (cell[:, 1] >= 0) & (cell[:, 0] < g["shp"][0]) & (cell[:, 1] < g["shp"][1])
        lvl = (P[:, 2] > z0 - 0.3) & (P[:, 2] < z0 + 3.0)
        info = {"scene": scene, "z_floor": z0, "n_rooms": len(g["keep"]), "room_areas_m2": [g["areas"][r] for r in g["keep"]],
                "adjacency": {str(r): g["adj"][r] for r in g["keep"]}, "steps": []}
        fig, axes = plt.subplots(1, 2, figsize=(13, 6))
        ax = axes[0]
        ax.imshow(np.where(g["floor"], 0.85, 1.0).T, origin="lower", cmap="gray", vmin=0, vmax=1,
                  extent=[g["xy0"][0], g["xy0"][0] + g["shp"][0] * CELL, g["xy0"][1], g["xy0"][1] + g["shp"][1] * CELL])
        rm = np.ma.masked_where(~np.isin(g["rooms"], g["keep"]), g["rooms"])
        ax.imshow(rm.T, origin="lower", cmap="tab20", alpha=0.45,
                  extent=[g["xy0"][0], g["xy0"][0] + g["shp"][0] * CELL, g["xy0"][1], g["xy0"][1] + g["shp"][1] * CELL])
        wpts = P[(L == WALL) & lvl]; ax.scatter(wpts[::20, 0], wpts[::20, 1], s=0.1, c="#333333")
        for r in g["keep"]:
            c = np.argwhere(g["rooms"] == r).mean(0) * CELL + g["xy0"]
            ax.text(c[0], c[1], "%d\n%.0f m²" % (g["keep"].index(r) + 1, g["areas"][r]), ha="center", va="center", fontsize=7)
        ax.set_aspect("equal"); ax.set_title("%s：部屋 %d（色）、壁の点（黒）、床の格子（灰）" % (scene, len(g["keep"])), loc="left", fontsize=9)
        for n, st in enumerate(steps, 1):
            region = ndi.binary_dilation(np.isin(g["rooms"], st), iterations=3)
            sel = np.zeros(len(P), bool); sel[inb] = region[cell[inb, 0], cell[inb, 1]]; sel &= lvl
            wsel = sel & (L == WALL)
            segs = lines_from_walls_v2(P[wsel] - [0, 0, z0], N[wsel])
            d = os.path.join(D, "BIM", "%s_n%d" % (scene, n)); os.makedirs(d, exist_ok=True)
            np.savetxt(os.path.join(d, "bimLineSeg.txt"), segs, fmt="%.4f")
            W = P[wsel] - [0, 0, z0]; write_pcd(os.path.join(d, "bim.pcd"), W[::4])
            flat = np.c_[W[:, :2], np.zeros(len(W))]
            key = np.unique(np.floor(flat[:, :2] / 0.05).astype(np.int64), axis=0, return_index=True)[1]
            write_pcd(os.path.join(d, "bim2d.pcd"), flat[np.sort(key)])
            json.dump({"z_floor": z0, "rooms": [g["keep"].index(r) + 1 for r in st], "n_segments": int(len(segs)),
                       "area_m2": float(sum(g["areas"][r] for r in st))}, open(os.path.join(d, "meta.json"), "w"))
            info["steps"].append({"n": n, "rooms": [g["keep"].index(r) + 1 for r in st], "area_m2": float(sum(g["areas"][r] for r in st)),
                                  "n_points": int(sel.sum()), "n_segments": int(len(segs)),
                                  "segment_length_total_m": float(np.sum(np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1]))) if len(segs) else 0.0})
        last = steps[-1] if steps else []
        L_all = np.loadtxt(os.path.join(D, "BIM", "%s_n%d" % (scene, len(steps)), "bimLineSeg.txt")).reshape(-1, 4) if steps else np.zeros((0, 4))
        ax = axes[1]; ax.scatter(wpts[::20, 0], wpts[::20, 1], s=0.1, c="#bbbbbb")
        for x in L_all:
            ax.plot([x[0], x[2]], [x[1], x[3]], color="#d0021b", lw=1.2)
        ax.set_aspect("equal"); ax.set_title("全部の部屋の参照から作った BIM 線分（%d 本）" % len(L_all), loc="left", fontsize=9)
        fig.tight_layout(); fig.savefig(os.path.join(D, "fig", scene + ".png"), dpi=100); plt.close(fig)
        json.dump(info, open(os.path.join(D, "rooms", scene + ".json"), "w"), indent=1, ensure_ascii=False)
        print(scene, "z0 %.2f rooms %d areas %s" % (z0, len(g["keep"]), [round(a) for a in info["room_areas_m2"]]),
              "steps", [(s["n"], round(s["area_m2"]), s["n_segments"]) for s in info["steps"]], flush=True)


if __name__ == "__main__":
    main()
