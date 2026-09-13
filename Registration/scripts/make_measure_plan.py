"""R24 §1 — 現地で測る位置を決め、平面図に落とす。

なぜ要るか
----------
`measure_spec.json` の `at` は BIM 座標である。**現地でその座標は分からない。**
そこで **BIM の平面図に測定位置を描き、扉を目印にして「この位置」と示せるようにする。**

測定位置の選び方（**先に決めて固定する。結果を見てから選び直さない**）：

1. 室の内部で、壁から 1.0 m 以上離れている
2. **その位置の半径 1.0 m に、スキャンの床点と天井点が両方 200 点以上ある**
   （スキャンが届いていない場所を指定しても測れないため）
3. 互いに 2.0 m 以上離す（同じ場所を2回測っても分布が出ない）

    conda activate sni-slam
    python Registration/scripts/make_measure_plan.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt      # noqa: E402
import numpy as np                   # noqa: E402
import yaml                          # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "Registration"))
sys.path.insert(0, os.path.join(REPO, "Registration", "scripts"))
os.chdir(REPO)

from failure_decomposition import provenance          # noqa: E402
from regbim import io_utils, metrics                  # noqa: E402
from regbim.labels import NAME_TO_ID                  # noqa: E402

MIN_PTS = 200          # 半径内に必要なスキャン点数（床・天井それぞれ）
WALL_CLEAR = 1.0       # 壁からの最小距離 [m]
SEP = 5.0              # 測定位置どうしの最小間隔 [m]。室の端から端を含めたいので広く取る
RADIUS = 1.0

# 図のラベルは英字にする。この環境に日本語フォントが無く、
# 日本語を入れると豆腐（□）になって現地で読めないため。


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default="m3_block_b")
    ap.add_argument("--room", default="411", choices=["410", "411"])
    ap.add_argument("--n-spots", type=int, default=3)
    ap.add_argument("--out", default="output/review_2026-09-13")
    args = ap.parse_args()

    cond = "E1" if args.room == "410" else "E2"
    cfg = yaml.safe_load(open("Registration/configs/realdata/%s__%s.yaml"
                              % (args.scene, cond)))
    cfg["source"] = dict(cfg["source"], seed=0)
    src = io_utils.load_source_cloud(cfg)
    dst = io_utils.load_reference_cloud(cfg)
    G = np.asarray(json.load(open("output/GT_alignment/T_gt/T_gt_%s.json"
                                  % args.scene))["T_gt"],
                   dtype=np.float64).reshape(4, 4)
    sp = metrics.apply_sim3(G, src.points)      # スキャンを BIM 座標へ

    FL, CE, WA, DR = (NAME_TO_ID[k] for k in ("floor", "ceiling", "wall", "door"))
    bim_fl = dst.points[dst.labels == FL]
    bim_wa = dst.points[dst.labels == WA]
    bim_dr = dst.points[dst.labels == DR]
    s_fl = sp[src.labels == FL]
    s_ce = sp[src.labels == CE]
    print("BIM: 床 %d / 壁 %d / 扉 %d 点   スキャン: 床 %d / 天井 %d 点"
          % (len(bim_fl), len(bim_wa), len(bim_dr), len(s_fl), len(s_ce)))

    from scipy.spatial import cKDTree
    t_fl, t_ce = cKDTree(s_fl[:, :2]), cKDTree(s_ce[:, :2])
    t_wa = cKDTree(bim_wa[:, :2])
    t_bimfl = cKDTree(bim_fl[:, :2])

    # 候補格子（BIM の床がある場所のみ）
    lo, hi = bim_fl[:, :2].min(0), bim_fl[:, :2].max(0)
    gx = np.arange(lo[0], hi[0], 0.25)
    gy = np.arange(lo[1], hi[1], 0.25)
    GX, GY = np.meshgrid(gx, gy)
    cand = np.column_stack([GX.ravel(), GY.ravel()])
    cand = cand[t_bimfl.query(cand)[0] < 0.2]                  # 床の上
    cand = cand[t_wa.query(cand)[0] > WALL_CLEAR]              # 壁から離す
    n_fl = np.array([len(t_fl.query_ball_point(p, RADIUS)) for p in cand])
    n_ce = np.array([len(t_ce.query_ball_point(p, RADIUS)) for p in cand])
    ok = (n_fl >= MIN_PTS) & (n_ce >= MIN_PTS)
    print("候補格子 %d 点 → スキャンが十分な点 %d 個" % (len(cand), int(ok.sum())))
    cand, n_fl, n_ce = cand[ok], n_fl[ok], n_ce[ok]

    # **室全体に散らす**（最遠点サンプリング）。同じ隅に固まると、
    # 天井が水平でない可能性を見るという目的を果たせない。
    # 1つ目はスキャン点が最も多い場所、以降は既選択から最も遠い場所。
    spots = []
    i0 = int(np.argmax(np.minimum(n_fl, n_ce)))
    spots.append({"xy": cand[i0], "n_floor": int(n_fl[i0]), "n_ceiling": int(n_ce[i0])})
    while len(spots) < args.n_spots:
        d = np.min([np.linalg.norm(cand - s["xy"], axis=1) for s in spots], axis=0)
        i = int(np.argmax(d))
        if d[i] < SEP:
            print("  （これ以上 %.1f m 離した位置が取れない。%d 個で止める）"
                  % (SEP, len(spots)))
            break
        spots.append({"xy": cand[i], "n_floor": int(n_fl[i]), "n_ceiling": int(n_ce[i])})
    print("\n選んだ測定位置：")
    for i, s in enumerate(spots, 1):
        print("  M%d  x=%+.2f  y=%+.2f   半径 1 m 内のスキャン点：床 %d / 天井 %d"
              % (i, s["xy"][0], s["xy"][1], s["n_floor"], s["n_ceiling"]))

    # ---- 平面図 ----
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.scatter(bim_fl[:, 0], bim_fl[:, 1], s=1, c="#dddddd", label="BIM floor")
    ax.scatter(bim_wa[:, 0], bim_wa[:, 1], s=2, c="#555555", label="BIM wall")
    if len(bim_dr):
        ax.scatter(bim_dr[:, 0], bim_dr[:, 1], s=28, c="#e02020",
                   label="BIM door (landmark)")
    ax.scatter(s_fl[::20, 0], s_fl[::20, 1], s=1, c="#ffa500", alpha=0.35,
               label="scan floor")
    for i, s in enumerate(spots, 1):
        ax.add_patch(plt.Circle(s["xy"], RADIUS, fill=False, color="#0070c0", lw=1.6))
        ax.plot(*s["xy"], marker="x", ms=11, mew=2.5, color="#0070c0")
        ax.annotate("M%d  (%+.2f, %+.2f)" % (i, s["xy"][0], s["xy"][1]),
                    s["xy"], textcoords="offset points", xytext=(10, 8),
                    fontsize=11, color="#0070c0", weight="bold")
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]  (BIM coordinates)")
    ax.set_ylabel("y [m]  (BIM coordinates)")
    ax.set_title("Room %s - plan view and measurement spots (circle = 1 m radius)"
                 % args.room)
    ax.legend(loc="lower left", fontsize=9, markerscale=4, framealpha=0.95)
    ax.grid(alpha=0.3)
    os.makedirs(args.out, exist_ok=True)
    png = os.path.join(args.out, "measure_plan_%s.png" % args.room)
    fig.tight_layout()
    fig.savefig(png, dpi=150)
    print("\n平面図: %s" % png)

    js = os.path.join(args.out, "measure_plan_%s.json" % args.room)
    with open(js, "w") as f:
        json.dump({"provenance": provenance(), "room": args.room, "scene": args.scene,
                   "config": "Registration/configs/realdata/%s__%s.yaml" % (args.scene, cond),
                   "criteria": {"min_scan_points": MIN_PTS, "wall_clearance_m": WALL_CLEAR,
                                "separation_m": SEP, "radius_m": RADIUS},
                   "spots": [{"id": "M%d" % (i + 1), "x": float(s["xy"][0]),
                              "y": float(s["xy"][1]), "n_floor": s["n_floor"],
                              "n_ceiling": s["n_ceiling"]}
                             for i, s in enumerate(spots)]}, f, indent=2, ensure_ascii=False)
    print("測定位置: %s" % js)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
