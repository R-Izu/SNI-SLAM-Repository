"""R34 §1-1 — **参照の壁面が、ラスタ化の格子境界に乗っていないか。**

なぜ確かめるか（R34 §1）
------------------------
**source は $s_j R_k$ で任意に変換されるので、セル境界に乗る確率は実質ゼロである。**
**しかし参照はそのまま格子に載る。**
**BIM の壁が設計上の丸い座標に置かれていれば、参照側で同じ縮退が起きうる。**
**「測度ゼロの偶然」は source には当てはまるが、参照には当てはまらないかもしれない。**

出すもの
--------
1. 壁の主要な面の座標（ラスタ化に使う座標系で）
2. 各面の座標を 0.25 m で割った余り
3. 余りが 0 または 0.25 の近く（±0.01 m）にある面の枚数
4. **ラスタ化の原点**（格子の位相）
"""

import os
import sys

import numpy as np
import yaml

sys.path.insert(0, "/home/student/rizu/SNI-SLAM/Registration")
sys.path.insert(0, "/home/student/rizu/SNI-SLAM/Registration/scripts")
os.chdir("/home/student/rizu/SNI-SLAM")

from regbim import io_utils, plan_correlate as pc     # noqa: E402
from regbim.labels import NAME_TO_ID                  # noqa: E402

CELL = pc.DEFAULTS["cell_m"]
TOL = 0.01          # 「境界の近く」とみなす幅（R34 §1-1）


def faces(xy_1d, min_pts=200, bin_w=0.02):
    """1 次元に並べた座標から、点が集中する位置（＝面）を拾う。"""
    if len(xy_1d) == 0:
        return []
    lo, hi = xy_1d.min(), xy_1d.max()
    nb = max(int(np.ceil((hi - lo) / bin_w)), 1)
    h, edges = np.histogram(xy_1d, bins=nb, range=(lo, lo + nb * bin_w))
    out = []
    for i in np.argsort(-h):
        if h[i] < min_pts:
            break
        c = 0.5 * (edges[i] + edges[i + 1])
        if all(abs(c - p) > 0.3 for p, _ in out):     # 30 cm 以内は同じ面
            out.append((float(c), int(h[i])))
    return sorted(out)


def report(name, cfg_path):
    cfg = yaml.safe_load(open(cfg_path))
    dst = io_utils.load_reference_cloud(cfg)
    wall = NAME_TO_ID["wall"]
    wm = dst.labels == wall
    wx, wy = pc.wall_groups(dst.points[wm], dst.normals[wm])

    # ラスタ化の原点（horizontal_candidates と同じ決め方）
    allxy = np.vstack([v for v in (wx, wy) if len(v)])
    origin = np.floor(allxy.min(axis=0) / CELL) * CELL

    print("\n" + "=" * 70)
    print("## %s" % name)
    print("   %s" % cfg_path)
    print("   参照の壁 %d 点（x 法線 %d / y 法線 %d）" % (wm.sum(), len(wx), len(wy)))
    print("   **ラスタ化の原点 = (%.4f, %.4f)**（セル幅 %.2f m の整数倍に丸めて決まる）"
          % (origin[0], origin[1], CELL))
    print("   参照の広がり x %.2f m / y %.2f m"
          % (allxy[:, 0].ptp(), allxy[:, 1].ptp()))

    n_on = n_all = 0
    for axis, (pts, ax_i) in (("x", (wx, 0)), ("y", (wy, 1))):
        fs = faces(pts[:, ax_i]) if len(pts) else []
        if not fs:
            continue
        print("\n   %s 法線の面（%d 枚）" % (axis, len(fs)))
        print("     %10s %10s %12s %10s %s"
              % ("座標[m]", "点数", "原点からの差", "余り[m]", "境界上か"))
        for c, cnt in fs:
            rel = c - origin[ax_i]
            rem = rel % CELL
            d_edge = min(rem, CELL - rem)          # 最寄りのセル境界までの距離
            on = d_edge <= TOL
            n_all += 1
            n_on += int(on)
            print("     %10.4f %10d %12.4f %10.4f %s"
                  % (c, cnt, rel, rem, "**乗る**" if on else "乗らない"))
    print("\n   **境界上（±%.2f m）の面: %d / %d 枚**" % (TOL, n_on, n_all))
    return n_on, n_all


tot_on = tot_all = 0
for nm, p in (("実データの参照（IFC・室 411＋410）",
               "Registration/configs/realdata/m3_block_b__E2.yaml"),
              ("GT-A の参照（Replica room_0）",
               "Registration/configs/gt_a/room_0.yaml")):
    try:
        a, b = report(nm, p)
        tot_on += a
        tot_all += b
    except Exception as ex:
        print("\n## %s: 読めない（%s: %s）" % (nm, type(ex).__name__, ex))

print("\n" + "=" * 70)
print("**合計 %d / %d 枚が境界上**" % (tot_on, tot_all))
print("R34 §1-2 の分岐：")
print("  境界に乗る面がほとんど無い → §5 の正当化が成立。このまま進む")
print("  境界に乗る面が相当数ある   → 正当化は成立しない。頑健化が要る")
