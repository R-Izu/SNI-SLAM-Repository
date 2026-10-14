"""R48 §6 事後の確認 c（libim の環境）：frl_apartment_0〜5 の間取りを比べる。**事後。登録は回さない。**

- 図：6 シーンの平面図（主の階の床の上 0.3〜1.8 m の壁・扉の点、全部の部屋の参照の v2 の BIM 線分）を同じ縮尺で並べる
- 重なりの数：シーンの組ごとに、全部の部屋の参照の BIM 線分を 0.05 m おきの点にし、対称の chamfer（平均の最近傍距離の平均）
  (i) Replica の座標のまま、(ii) 2D の剛体で合わせた後（ヨー 0〜359°、1° おきに重心を合わせてから 2D ICP 30 回、chamfer が最小のもの）
- 同じく、壁の点（0.05 m の格子に間引き）どうしでも (i)・(ii)
出力：`~/rizu/r48_data/posthoc/c_frl.png`・`c.json`
"""
import json, os, sys
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r48_common as C                                          # noqa: E402
import matplotlib; matplotlib.use("Agg")                        # noqa: E402
import matplotlib.pyplot as plt                                 # noqa: E402
from matplotlib import font_manager as fm                       # noqa: E402
for f in ("/mnt/c/Windows/Fonts/arial.ttf", "/home/student/plotconda/meiryo_ui.ttf"):
    fm.fontManager.addfont(f)
plt.rcParams["font.family"] = ["Arial", "Meiryo UI"]; plt.rcParams["font.size"] = 8

SC = ["frl_apartment_%d" % i for i in range(6)]
OUT = os.path.join(C.D, "posthoc"); os.makedirs(OUT, exist_ok=True)


def last_step(s):
    k = 1
    while os.path.exists(os.path.join(C.D, "BIM", "%s_n%d" % (s, k + 1))):
        k += 1
    return "%s_n%d" % (s, k)


def seg_points(segs, step=0.05):
    out = []
    for x in segs:
        a, b = x[:2], x[2:4]; n = max(2, int(np.ceil(np.linalg.norm(b - a) / step)) + 1)
        out.append(a + np.linspace(0, 1, n)[:, None] * (b - a))
    return np.concatenate(out)


def chamfer(A, B):
    return float((cKDTree(B).query(A)[0].mean() + cKDTree(A).query(B)[0].mean()) / 2)


def rot(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s], [s, c]])


def align(A, B):
    """A を 2D の剛体で B に合わせる。chamfer が最小の (chamfer, ヨー[°], 並進)"""
    tb = cKDTree(B); best = (np.inf, 0.0, np.zeros(2))
    for yd in range(0, 360):
        R = rot(np.radians(yd)); X = A @ R.T; t = B.mean(0) - X.mean(0); Rc, tc = R, t
        for _ in range(30):
            Y = X @ np.eye(2).T + t; idx = tb.query(Y)[1]; M = B[idx]
            mu_y, mu_m = Y.mean(0), M.mean(0); H = (Y - mu_y).T @ (M - mu_m); U, _, Vt = np.linalg.svd(H)
            dR = Vt.T @ U.T
            if np.linalg.det(dR) < 0:
                Vt[-1] *= -1; dR = Vt.T @ U.T
            X = (X + t - mu_y) @ dR.T + mu_m - t; Rc = dR @ Rc
        Y = X + t; c = chamfer(Y, B)
        if c < best[0]:
            ang = float(np.degrees(np.arctan2(Rc[1, 0], Rc[0, 0])))
            best = (c, ang, (Y - A @ Rc.T).mean(0))
    return best


def main():
    data = {}
    for s in SC:
        ref = last_step(s); segs = np.loadtxt(os.path.join(C.D, "BIM", ref, "bimLineSeg.txt")).reshape(-1, 4)
        z = np.load(os.path.join(C.D, "ref", s + ".npz")); P, L = z["points"].astype(float), z["labels"]
        z0 = json.load(open(os.path.join(C.D, "rooms", s + ".json")))["z_floor"]
        W = P[np.isin(L, [C.WALL, C.DOOR]) & (P[:, 2] > z0 + 0.3) & (P[:, 2] < z0 + 1.8), :2]
        W = W[np.unique(np.floor(W / 0.05).astype(np.int64), axis=0, return_index=True)[1]]
        info = json.load(open(os.path.join(C.D, "rooms", s + ".json")))
        data[s] = {"ref": ref, "segs": segs, "lp": seg_points(segs), "W": W, "areas": info["room_areas_m2"], "z0": z0}
    res = {"scenes": {s: {"ref": d["ref"], "n_segments": int(len(d["segs"])), "room_areas_m2": d["areas"], "z_floor": d["z0"],
                          "xy_min": d["W"].min(0).tolist(), "xy_max": d["W"].max(0).tolist()} for s, d in data.items()}, "pairs": {}}
    for i in range(6):
        for j in range(i + 1, 6):
            a, b = data[SC[i]], data[SC[j]]
            cl_raw = chamfer(a["lp"], b["lp"]); cw_raw = chamfer(a["W"], b["W"])
            cl_al = align(a["lp"], b["lp"])
            Wa = a["W"] @ rot(np.radians(cl_al[1])).T + cl_al[2]; cw_al = chamfer(Wa, b["W"])
            res["pairs"]["%s|%s" % (SC[i], SC[j])] = {"lines_chamfer_raw_m": cl_raw, "walls_chamfer_raw_m": cw_raw,
                                                      "lines_chamfer_aligned_m": cl_al[0], "aligned_yaw_deg": cl_al[1],
                                                      "aligned_t_xy": [float(x) for x in cl_al[2]], "walls_chamfer_aligned_m": cw_al}
            print(SC[i], SC[j], "lines raw %.3f aligned %.3f (yaw %.1f) | walls raw %.3f aligned %.3f" % (cl_raw, cl_al[0], cl_al[1], cw_raw, cw_al), flush=True)
    fig, axs = plt.subplots(2, 3, figsize=(15, 10), sharex=True, sharey=True)
    for ax, s in zip(axs.ravel(), SC):
        d = data[s]
        ax.scatter(d["W"][:, 0], d["W"][:, 1], s=0.2, c="#888888")
        for x in d["segs"]:
            ax.plot([x[0], x[2]], [x[1], x[3]], color="#d0021b", lw=1.2)
        ax.set_aspect("equal"); ax.set_title("%s：部屋 %s m²、線分 %d 本（%s）" % (s, "・".join("%.1f" % a for a in d["areas"]), len(d["segs"]), d["ref"]),
                                             loc="left", fontsize=9)
    fig.suptitle("frl_apartment_0〜5：主の階の壁・扉の点（灰、床の上 0.3〜1.8 m）と全部の部屋の参照の BIM 線分（赤）。Replica の座標のまま", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(os.path.join(OUT, "c_frl.png"), dpi=80); plt.close(fig)
    json.dump(res, open(os.path.join(OUT, "c.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
