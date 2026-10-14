"""R48 §6 事後の確認 a（libim の環境）：apartment_2_n1 の参照の上面図と、案A の失敗 25 件の推定位置。**事後。登録は回さない。**

図（3 枚）
1. 部分参照の点をクラスごとに（壁・扉・窓・床、上面）、v2 の BIM 線分、線分の交点（BIM 側の角の位置の目安）
2. シーン全体の参照の壁・扉・窓の点（主の階、床の上 0.3〜1.8 m）と、部屋分けの結果、部分参照の範囲
3. 案A の失敗 25 件：source（その入力の範囲）の構造点を T_est · P で参照の座標に置いたもの（失敗ごとに色）と、正解 G で置いたもの（黒）
数：参照の壁の点の向き・位置の刻みの数（`lines_from_walls_v2` の内部）、範囲ごとの壁の点の数
出力：`~/rizu/r48_data/posthoc/a_*.png`・`a.json`
"""
import json, os, sys
import numpy as np
from scipy import ndimage as ndi
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r48_common as C                                          # noqa: E402
import r48_rooms as rr                                          # noqa: E402
import matplotlib; matplotlib.use("Agg")                        # noqa: E402
import matplotlib.pyplot as plt                                 # noqa: E402
from matplotlib import font_manager as fm                       # noqa: E402
for f in ("/mnt/c/Windows/Fonts/arial.ttf", "/home/student/plotconda/meiryo_ui.ttf"):
    fm.fontManager.addfont(f)
plt.rcParams["font.family"] = ["Arial", "Meiryo UI"]; plt.rcParams["font.size"] = 8

REF = "apartment_2_n1"; SCENE = "apartment_2"
OUT = os.path.join(C.D, "posthoc"); os.makedirs(OUT, exist_ok=True)
COL = {C.WALL: "#333333", C.DOOR: "#e08a00", C.WINDOW: "#2a6fdb", C.FLOOR: "#cccccc", C.CEIL: "#9b59b6"}
NAME = {C.WALL: "壁", C.DOOR: "扉", C.WINDOW: "窓", C.FLOOR: "床", C.CEIL: "天井"}


def intersections(segs):
    pts = []
    for i in range(len(segs)):
        for j in range(i + 1, len(segs)):
            a, b = segs[i, :2], segs[i, 2:]; c, d = segs[j, :2], segs[j, 2:]
            A = np.array([b - a, c - d]).T
            if abs(np.linalg.det(A)) < 1e-9:
                continue
            s, t = np.linalg.solve(A, c - a)
            pts.append(a + s * (b - a))
    return np.asarray(pts).reshape(-1, 2)


def main():
    out = {}
    z = np.load(os.path.join(C.D, "refs", REF + ".npz")); P, L, N = z["points"].astype(float), z["labels"].astype(int), z["normals"].astype(float)
    z0 = json.load(open(os.path.join(C.D, "BIM", REF, "meta.json")))["z_floor"]
    segs = np.loadtxt(os.path.join(C.D, "BIM", REF, "bimLineSeg.txt")).reshape(-1, 4)
    out["class_counts"] = {NAME[k]: int((L == k).sum()) for k in NAME}
    band = (P[:, 2] > z0 + 0.3) & (P[:, 2] < z0 + 1.8)
    out["class_counts_0p3_1p8"] = {NAME[k]: int(((L == k) & band).sum()) for k in NAME}
    # v2 の内部の刻み
    h = (L == C.WALL) & (np.abs(N[:, 2]) < 0.3); Pw, Nw = P[h], N[h]
    ang = np.degrees(np.arctan2(Nw[:, 1], Nw[:, 0])) % 180.0; ab = np.minimum(ang.astype(int), 179)
    cnt = np.bincount(ab, minlength=180); thr = max(50, 0.005 * len(Pw))
    groups = []
    for grp in rr.merge_runs(np.flatnonzero(cnt >= thr), n_wrap=180):
        m = np.isin(ab, grp); a = ang[m]; a = (a - a[0] + 90) % 180 - 90 + a[0]
        th = np.radians(np.median(a)); n = np.array([np.cos(th), np.sin(th)])
        off = Pw[m, :2] @ n; ob = np.floor(off / 0.05).astype(int); u, c = np.unique(ob, return_counts=True)
        groups.append({"angle_bins": [int(grp[0]), int(grp[-1])], "n_points": int(m.sum()), "normal_deg": float(np.degrees(th)),
                       "offset_bins_ge50": [[round(float(b) * 0.05, 2), int(k)] for b, k in zip(u, c) if k >= 50]})
    out["v2_angle_threshold"] = float(thr); out["v2_groups"] = groups
    out["segments"] = segs.tolist(); X = intersections(segs); out["segment_intersections"] = X.tolist()
    # シーン全体
    g = rr.segment(SCENE); SP, SL = g["P"], g["L"]
    lvl = (SP[:, 2] > g["z0"] + 0.3) & (SP[:, 2] < g["z0"] + 1.8)
    # 案A の失敗
    sc = json.load(open(os.path.expanduser("~/rizu/SNI-SLAM/Registration/output/diag/r48/r48_score.json")))
    man = json.load(open(os.path.join(C.D, "inputs_manifest.json"))); G = np.asarray(man["G"])
    fails = []
    for cut in ("0p5", "3", "inf"):
        cell = sc["refs"][REF]["cells"][cut]
        rows = json.load(open(os.path.join(C.D, "results_sni", "%s__%s.json" % (REF, cut))))["rows"]
        src = np.load(os.path.join(C.D, "inputs_sni", "%s__%s.npz" % (REF, cut)))
        S = src["points"].astype(float)[np.isin(src["labels"], [C.WALL, C.FLOOR, C.CEIL])]
        S = S[np.linspace(0, len(S) - 1, 4000).astype(int)]
        for k, f in enumerate(cell["fails"]["planA"]):
            if f:
                T = np.asarray(rows["planA|%d" % k]["T_est"]) @ np.asarray(rows["planA|%d" % k]["P"])
                Q = C.apply(T, S); Qg = C.apply(G, S)
                fails.append({"cut": cut, "trial": k, "eR": cell["per"]["planA"][k]["eR"], "eO": cell["per"]["planA"][k]["eO"],
                              "q": cell["per"]["planA"][k]["q"], "placed_centroid_xy": Q[:, :2].mean(0).tolist(),
                              "true_centroid_xy": Qg[:, :2].mean(0).tolist(), "_Q": Q, "_Qg": Qg})
    out["planA_fails"] = [{k: v for k, v in f.items() if not k.startswith("_")} for f in fails]
    # 図
    fig, axs = plt.subplots(1, 3, figsize=(18, 6.5))
    ax = axs[0]
    for k in (C.FLOOR, C.WINDOW, C.DOOR, C.WALL):
        m = L == k; s = P[m][::max(1, m.sum() // 15000)]
        ax.scatter(s[:, 0], s[:, 1], s=0.3 if k == C.FLOOR else 0.6, c=COL[k], label="%s（%d 点）" % (NAME[k], m.sum()))
    for x in segs:
        ax.plot([x[0], x[2]], [x[1], x[3]], color="#d0021b", lw=2.2)
    if len(X):
        ax.plot(X[:, 0], X[:, 1], "o", ms=9, mfc="none", mec="#d0021b", mew=2)
    ax.set_aspect("equal"); ax.legend(loc="upper left", markerscale=12, fontsize=7)
    ax.set_title("①%s の参照（上面）。赤：v2 の BIM 線分 %d 本、○：線分の交点 %d 個" % (REF, len(segs), len(X)), loc="left", fontsize=9)
    ax = axs[1]
    ax.imshow(np.ma.masked_where(~np.isin(g["rooms"], g["keep"]), g["rooms"]).T, origin="lower", cmap="tab20", alpha=0.35,
              extent=[g["xy0"][0], g["xy0"][0] + g["shp"][0] * rr.CELL, g["xy0"][1], g["xy0"][1] + g["shp"][1] * rr.CELL])
    for k in (C.WINDOW, C.DOOR, C.WALL):
        m = (SL == k) & lvl; s = SP[m][::max(1, m.sum() // 20000)]
        ax.scatter(s[:, 0], s[:, 1], s=0.3, c=COL[k], label=NAME[k])
    for x in segs:
        ax.plot([x[0], x[2]], [x[1], x[3]], color="#d0021b", lw=1.5)
    ax.set_aspect("equal"); ax.legend(loc="upper left", markerscale=15, fontsize=7)
    ax.set_title("②シーン全体の壁・扉・窓（床の上 0.3〜1.8 m）と部屋分け（色）", loc="left", fontsize=9)
    ax = axs[2]
    m = np.isin(SL, [C.WALL, C.DOOR, C.WINDOW]) & lvl; s = SP[m][::max(1, m.sum() // 20000)]
    ax.scatter(s[:, 0], s[:, 1], s=0.3, c="#bbbbbb", label="シーン全体の壁・扉・窓")
    mode = lambda e: "e_R≈90°" if 45 <= e < 135 else ("e_R≈180°" if e >= 135 else "e_R<45°")
    mc = {"e_R≈90°": "#2a6fdb", "e_R≈180°": "#d94a2b", "e_R<45°": "#2ca02c"}
    shown = set()
    for f in fails:
        md = mode(f["eR"]); mk = "o" if f["cut"] == "3" else "s"
        if md not in shown:
            ax.scatter(f["_Q"][::6, 0], f["_Q"][::6, 1], s=0.4, color=mc[md], alpha=0.5)
            shown.add(md)
        ax.plot(*f["placed_centroid_xy"], mk, ms=7, mfc="none", mec=mc[md], mew=1.5)
    for cut, mk in (("3", "o"), ("inf", "s")):
        gt = [f for f in fails if f["cut"] == cut][0]
        ax.plot(*gt["true_centroid_xy"], mk, ms=11, color="k")
    ax.scatter(fails[0]["_Qg"][::6, 0], fails[0]["_Qg"][::6, 1], s=0.4, c="k", alpha=0.4)
    for md, c in mc.items():
        ax.plot([], [], "o", color=c, label="%s（%d 件）" % (md, sum(mode(f["eR"]) == md for f in fails)))
    ax.plot([], [], "o", mfc="none", mec="k", label="○ r = 3 m、□ 切らない（推定の重心）")
    ax.plot([], [], "o", color="k", label="黒：正解の位置（重心と点）")
    ax.set_aspect("equal"); ax.legend(loc="lower left", fontsize=7, markerscale=1.5)
    ax.set_title("③案A の失敗 %d 件：source の構造点を推定で置いた重心と、向きごとの例 1 件の点" % len(fails), loc="left", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "a_%s.png" % REF), dpi=90); plt.close(fig)
    json.dump(out, open(os.path.join(OUT, "a.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in out.items() if k != "planA_fails"}, ensure_ascii=False)[:3000])
    for f in out["planA_fails"]:
        print(f["cut"], f["trial"], "eR %.1f eO %.2f" % (f["eR"], f["eO"]), "placed", np.round(f["placed_centroid_xy"], 2), "true", np.round(f["true_centroid_xy"], 2))


if __name__ == "__main__":
    main()
