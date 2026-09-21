"""R33 §5-1 の3量を出す。

1. 真の縮尺に最も近い格子点の相対誤差（N=170 で）
2. その格子点での一致セル数と、勝った候補の一致セル数
3. 相関部分の実時間（**見積もりで代用しない**）
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, "/home/student/rizu/SNI-SLAM/Registration")
sys.path.insert(0, "/home/student/rizu/SNI-SLAM/Registration/tests")
os.chdir("/home/student/rizu/SNI-SLAM")

from regbim import plan_correlate as pc          # noqa: E402
from regbim.labels import NAME_TO_ID             # noqa: E402
from test_plan_candidates import box_room        # noqa: E402

p = pc.DEFAULTS
N = p["scale_n"]
scales = np.exp(np.linspace(np.log(p["scale_lo"]), np.log(p["scale_hi"]), N))
k = int(np.argmin(np.abs(scales - 1.0)))
print("## 1. 格子（N=%d）" % N)
print("  刻み %.4f%% / 半刻み %.4f%%" % (100 * (scales[1] / scales[0] - 1),
                                        100 * (scales[1] / scales[0] - 1) / 2))
print("  真値 1.0 に最も近い格子点 %.6f（**相対誤差 %.4f%%**）"
      % (scales[k], 100 * abs(scales[k] - 1.0)))
for L in (8.0, 18.2):
    print("    L=%.1f m での変位 %.1f mm（セル幅 %.0f mm）"
          % (L, 1000 * abs(scales[k] - 1.0) * L, 1000 * p["cell_m"]))

dst_p, dst_l, dst_n = box_room(0, 0, 8, 5)
room = box_room(0, 0, 8, 5)
cor = box_room(16, 0, 22, 2.2)
bp = np.vstack([room[0], cor[0]])
bl = np.concatenate([room[1], cor[1]])
bn = np.vstack([room[2], cor[2]])

wall = NAME_TO_ID["wall"]
sp, sl, sn = pc._subsample_by_role(bp, bl, bn, p["max_points"],
                                   p["subsample_seed"], wall,
                                   NAME_TO_ID["floor"], NAME_TO_ID["ceiling"])
dp, dl, dn = pc._subsample_by_role(dst_p, dst_l, dst_n, p["max_points"],
                                   p["subsample_seed"], wall,
                                   NAME_TO_ID["floor"], NAME_TO_ID["ceiling"])
dm, wm = dl == wall, sl == wall
dx, dy = pc.wall_groups(dp[dm], dn[dm])

print("\n## 2. 一致セル数の対比")
best = None
at_true = None
for i, s in enumerate(scales):
    sx, sy = pc.wall_groups(sp[wm] * s, sn[wm])
    hs = pc.horizontal_candidates(sx, sy, dx, dy, p["cell_m"], 1, p["peak_sep_m"])
    if not hs:
        continue
    sc = hs[0]["score"]
    if best is None or sc > best[1]:
        best = (s, sc, hs[0]["shift_xy"])
    if i == k:
        at_true = (s, sc, hs[0]["shift_xy"])
print("  真値に最も近い格子点 s=%.6f : 一致 %.0f セル（移動 %s）"
      % (at_true[0], at_true[1], np.round(at_true[2], 2).tolist()))
print("  **勝った候補**        s=%.6f : 一致 %.0f セル（移動 %s）"
      % (best[0], best[1], np.round(best[2], 2).tolist()))
print("  → 勝者は真値の %.1f 倍のスコア" % (best[1] / max(at_true[1], 1)))

print("\n## 3. 相関の実時間（4 向き × N × 2 群）")
rots = []
for j in range(4):
    a = j * np.pi / 2
    rots.append(np.array([[np.cos(a), -np.sin(a), 0],
                          [np.sin(a), np.cos(a), 0], [0, 0, 1.0]]))
t0 = time.time()
cands, diag = pc.generate_candidates(bp, bl, bn, dst_p, dst_l, dst_n,
                                     rots, NAME_TO_ID)
dt = time.time() - t0
print("  4 向き・N=%d の候補生成: **%.1f 秒**（相関 %d 回）"
      % (N, dt, 4 * N * 2))
print("  R29 §2-1 の予算 300 秒に対して %.1f%%" % (100 * dt / 300))
print("  候補 %d 個 / 生の候補 %d 個" % (len(cands), diag["n_raw"]))
