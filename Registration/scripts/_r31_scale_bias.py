"""相関値が縮尺に対してどう動くかを見る。

**仮説**：相関値を正規化していないので、**縮尺を小さくすると
source の壁セルが参照の狭い範囲に詰め込まれ、一致数が機械的に増える。**
そうなら、真値が 1.0 でも小さい縮尺が勝つ。

**これは設定値の調整の話ではなく、スコアの定義の話である。**
確かめてから報告する。**勝手に直さない**（R29 §7）。
"""

import os
import sys

import numpy as np

sys.path.insert(0, "/home/student/rizu/SNI-SLAM/Registration")
sys.path.insert(0, "/home/student/rizu/SNI-SLAM/Registration/tests")
os.chdir("/home/student/rizu/SNI-SLAM")

from regbim import plan_correlate as pc          # noqa: E402
from regbim.labels import NAME_TO_ID             # noqa: E402
from test_plan_candidates import box_room        # noqa: E402

dst_p, dst_l, dst_n = box_room(0, 0, 8, 5)
room = box_room(0, 0, 8, 5)
cor = box_room(16, 0, 22, 2.2)
base_p = np.vstack([room[0], cor[0]])
base_l = np.concatenate([room[1], cor[1]])
base_n = np.vstack([room[2], cor[2]])

p = pc.DEFAULTS
sp, sl, sn = pc._subsample(base_p, base_l, base_n, p["max_points"],
                           p["subsample_seed"])
dp, dl, dn = pc._subsample(dst_p, dst_l, dst_n, p["max_points"],
                           p["subsample_seed"])
wall = NAME_TO_ID["wall"]
dm = dl == wall
dx, dy = pc.wall_groups(dp[dm], dn[dm])
wm = sl == wall

print("%9s %12s %14s %12s"
      % ("縮尺", "最大相関値", "source 壁セル数", "正規化した値"))
print("-" * 52)
rows = []
for s in np.exp(np.linspace(np.log(0.6), np.log(1.6), 21)):
    sx, sy = pc.wall_groups(sp[wm] * s, sn[wm])
    hs = pc.horizontal_candidates(sx, sy, dx, dy, p["cell_m"],
                                  1, p["peak_sep_m"])
    if not hs:
        continue
    # source 側の占有セル数（縮尺で変わる。これが効いているかを見る）
    allxy = np.vstack([v for v in (sx, sy) if len(v)])
    ncell = len(np.unique(np.floor(allxy / p["cell_m"]).astype(np.int64), axis=0))
    sc = hs[0]["score"]
    rows.append((s, sc, ncell, sc / max(ncell, 1)))
    print("%9.4f %12.0f %14d %12.4f" % (s, sc, ncell, sc / max(ncell, 1)))

best_raw = max(rows, key=lambda r: r[1])
best_norm = max(rows, key=lambda r: r[3])
print("\n生の相関値が最大の縮尺   : %.4f" % best_raw[0])
print("正規化した値が最大の縮尺 : %.4f" % best_norm[0])
print("**真値は 1.0**")
print()
if abs(best_raw[0] - 1.0) > abs(best_norm[0] - 1.0):
    print("**仮説どおり：正規化しない相関値は小さい縮尺に偏る。**")
else:
    print("**仮説は支持されない。別の原因を探す。**")
