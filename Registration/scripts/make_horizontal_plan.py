"""R25 §5 — **水平**の面間距離を測る位置を、スキャンの被覆から先に探す。

なぜ要るか
----------
R24 で本人に頼んだ W1（南北の壁間 10.17 m）は、**スキャンが北の壁に届いていなかった**
（+y 側の面が 26 点）。**本人を無駄足にした。**
今度は**先にスキャン側で「両方の壁に十分な点がある位置」を探してから**渡す。

選定の条件（**先に決めて固定する。結果を見てから選び直さない**。R25 §5）
--------------------------------------------------------------------
1. 向かい合う2面**それぞれに 300 点以上**（室内高で使えた面は 440〜1078 点だった）
2. **できるだけ長い距離**を測る（同じ絶対誤差なら、長いほど相対誤差が小さい）
3. **現地で位置を特定できること**（壁からの距離で言えること＋平面図に印）
4. **複数の候補**を出す（什器で測れないときの代替）

**期待値（L_scan / L_bim）は現地へ渡す指示書には載せない。**
測る人が数値を知っていると、そこへ寄せてしまうため。**本報告には載せる。**

    conda activate sni-slam
    python Registration/scripts/make_horizontal_plan.py
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

from face_distance import AXES, fit_plane, select     # noqa: E402
from failure_decomposition import provenance          # noqa: E402
from regbim import io_utils, metrics                  # noqa: E402
from regbim.labels import NAME_TO_ID                  # noqa: E402

MIN_PTS = 300          # 各面に必要なスキャン点数（R25 §5）
RADIUS = 1.5           # 面に平行な方向の半径 [m]。メジャーを張る線の周り
Z_MEAS = 1.2           # 測定高さ [m]。室内高の測定と揃える
SEP = 3.0              # 候補どうしの最小間隔 [m]
GRID = 0.50            # 候補格子の間隔 [m]

# 図のラベルは英字。この環境に日本語フォントが無く、日本語は豆腐になる。


def evaluate(at, axis_key, src_pts, src_lab, src_nrm, dst, radius):
    """1 つの位置・1 つの方向について、向かい合う2面を当てはめる。"""
    ax = AXES["+" + axis_key][0]
    out = {}
    for side, nrm_key in (("a", "+" + axis_key), ("b", "-" + axis_key)):
        face = {"class": "wall", "normal": nrm_key}
        sgn = AXES[nrm_key][1]
        for who, pts, lab, nr in (("scan", src_pts, src_lab, src_nrm),
                                  ("bim", dst.points, dst.labels, dst.normals)):
            sel = select(pts, lab, face, at, radius, ax, nr)
            out["%s_%s" % (who, side)] = fit_plane(sel, ax, sgn, float(at[ax]))
    for who in ("scan", "bim"):
        a, b = out["%s_a" % who], out["%s_b" % who]
        out["L_%s_m" % who] = (None if not (a and b)
                               else float(abs(a["coord_m"] - b["coord_m"])))
    out["n_min"] = min([out["scan_%s" % s]["n_points"] if out["scan_%s" % s] else 0
                        for s in ("a", "b")])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default="m3_block_b")
    ap.add_argument("--config", default="Registration/configs/realdata/m3_block_b__E2.yaml")
    ap.add_argument("--gt-kit", default="output/GT_alignment")
    ap.add_argument("--n-spots", type=int, default=3)
    ap.add_argument("--out", default="output/review_2026-09-14")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    cfg["source"] = dict(cfg["source"], seed=0)
    src = io_utils.load_source_cloud(cfg)
    dst = io_utils.load_reference_cloud(cfg)
    G = np.asarray(json.load(open(os.path.join(
        args.gt_kit, "T_gt", "T_gt_%s.json" % args.scene)))["T_gt"],
        dtype=np.float64).reshape(4, 4)
    R_gt = metrics.decompose_sim3(G)[0]
    sp = metrics.apply_sim3(G, src.points)
    snr = None if src.normals is None else (np.asarray(src.normals) @ R_gt.T)

    FL, WA, DR = (NAME_TO_ID[k] for k in ("floor", "wall", "door"))
    bim_fl = dst.points[dst.labels == FL]
    bim_wa = dst.points[dst.labels == WA]
    bim_dr = dst.points[dst.labels == DR]

    from scipy.spatial import cKDTree
    t_bimfl = cKDTree(bim_fl[:, :2])
    lo, hi = bim_fl[:, :2].min(0), bim_fl[:, :2].max(0)
    gx = np.arange(lo[0], hi[0], GRID)
    gy = np.arange(lo[1], hi[1], GRID)
    GXm, GYm = np.meshgrid(gx, gy)
    cand = np.column_stack([GXm.ravel(), GYm.ravel()])
    cand = cand[t_bimfl.query(cand)[0] < 0.3]        # BIM の床の上だけ
    print("候補格子 %d 点（%.2f m 間隔、測定高さ z=%.2f m）" % (len(cand), GRID, Z_MEAS))

    rows = []
    for axis_key in ("x", "y"):
        for p in cand:
            at = np.array([p[0], p[1], Z_MEAS])
            r = evaluate(at, axis_key, sp, src.labels, snr, dst, RADIUS)
            if r["n_min"] >= MIN_PTS and r["L_scan_m"]:
                rows.append({"axis": axis_key, "at": at.tolist(), **r})
    print("両面 %d 点以上を満たす (位置, 方向) の組: %d" % (MIN_PTS, len(rows)))
    if not rows:
        print("**条件を満たす位置が無い。** MIN_PTS か RADIUS を見直すこと")
        return 1

    # 長い順に、**方向ごとに** SEP 以上離して採る。
    #
    # ★ 最初の実装は離れているかを 3 次元距離で見ていた。
    #   **X 方向の測定では、測線に沿った x 位置は選点に一切効かない**
    #   （面を選ぶ半径は測線に直交する y・z にしか掛からないため）。
    #   その結果 x だけ 3 m ずらした**同一の測定**が2件並んだ。
    #   離れているかは**測線に直交する座標だけ**で見る。
    #   これは値を良くするための変更ではなく、同じものを2回数えていた誤りの修正である。
    rows.sort(key=lambda r: -r["L_scan_m"])
    picked = []
    for axis_key in ("x", "y"):
        ax = AXES["+" + axis_key][0]
        other = [i for i in range(3) if i != ax]
        taken = [r for r in picked if r["axis"] == axis_key]
        for r in rows:
            if r["axis"] != axis_key:
                continue
            q = np.asarray(r["at"])[other]
            if all(np.linalg.norm(q - np.asarray(s["at"])[other]) >= SEP for s in taken):
                taken.append(r)
                picked.append(r)
            if len(taken) >= args.n_spots:
                break

    # 現地で言える形：BIM の壁の外形からの距離
    wlo, whi = bim_wa[:, :2].min(0), bim_wa[:, :2].max(0)
    print("\n選んだ測定位置（**期待値は現地の指示書には載せない**）：")
    spots = []
    for i, r in enumerate(picked, 1):
        x, y = r["at"][0], r["at"][1]
        desc = ("西端から %.1f m / 東端から %.1f m / 南端から %.1f m / 北端から %.1f m、"
                "床から %.1f m" % (x - wlo[0], whi[0] - x, y - wlo[1], whi[1] - y, Z_MEAS))
        sid = "H%d_411_%s" % (i, "width_ew" if r["axis"] == "x" else "width_ns")
        print("  %-18s %s 方向   位置 (%+.2f, %+.2f)" % (sid, r["axis"].upper(), x, y))
        print("     %s" % desc)
        print("     スキャン点数 +側 %d / −側 %d   L_scan %.4f m / L_bim %s"
              % (r["scan_a"]["n_points"], r["scan_b"]["n_points"], r["L_scan_m"],
                 "—" if r["L_bim_m"] is None else "%.4f m" % r["L_bim_m"]))
        spots.append({"id": sid, "axis": r["axis"], "at": r["at"], "where": desc,
                      "n_scan_plus": r["scan_a"]["n_points"],
                      "n_scan_minus": r["scan_b"]["n_points"],
                      "L_scan_m": r["L_scan_m"], "L_bim_m": r["L_bim_m"]})

    # ---- 平面図 ----
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.scatter(bim_fl[:, 0], bim_fl[:, 1], s=1, c="#dddddd", label="BIM floor")
    ax.scatter(bim_wa[:, 0], bim_wa[:, 1], s=2, c="#555555", label="BIM wall")
    if len(bim_dr):
        ax.scatter(bim_dr[:, 0], bim_dr[:, 1], s=28, c="#e02020",
                   label="BIM door (landmark)")
    s_wa = sp[src.labels == WA]
    ax.scatter(s_wa[::20, 0], s_wa[::20, 1], s=1, c="#ffa500", alpha=0.35,
               label="scan wall")
    for i, r in enumerate(picked, 1):
        x, y = r["at"][0], r["at"][1]
        a, b = r["scan_a"]["coord_m"], r["scan_b"]["coord_m"]
        if r["axis"] == "x":
            ax.annotate("", xy=(a, y), xytext=(b, y),
                        arrowprops=dict(arrowstyle="<->", color="#0070c0", lw=2.2))
        else:
            ax.annotate("", xy=(x, a), xytext=(x, b),
                        arrowprops=dict(arrowstyle="<->", color="#0070c0", lw=2.2))
        ax.plot(x, y, marker="x", ms=11, mew=2.5, color="#0070c0")
        ax.annotate("H%d (%s)" % (i, r["axis"].upper()), (x, y),
                    textcoords="offset points", xytext=(8, 8), fontsize=11,
                    color="#0070c0", weight="bold")
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]  (BIM coordinates)")
    ax.set_ylabel("y [m]  (BIM coordinates)")
    ax.set_title("Room 411 - horizontal face-to-face measurements (z = %.1f m)" % Z_MEAS)
    ax.legend(loc="upper left", fontsize=9, markerscale=4, framealpha=0.95)
    ax.grid(alpha=0.3)
    os.makedirs(args.out, exist_ok=True)
    png = os.path.join(args.out, "measure_plan_411_horizontal.png")
    fig.tight_layout()
    fig.savefig(png, dpi=150)
    print("\n平面図: %s" % png)

    js = os.path.join(args.out, "measure_plan_411_horizontal.json")
    with open(js, "w") as f:
        json.dump({"provenance": provenance(), "scene": args.scene, "config": args.config,
                   "gt_kit": args.gt_kit,
                   "criteria": {"min_points_per_face": MIN_PTS, "radius_m": RADIUS,
                                "z_m": Z_MEAS, "separation_m": SEP, "grid_m": GRID},
                   "n_feasible": len(rows), "spots": spots}, f, indent=2,
                  ensure_ascii=False)
    print("測定位置: %s" % js)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
