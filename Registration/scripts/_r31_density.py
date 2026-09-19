"""点密度が足りていないのではないか、を確かめる。

正解の縮尺 1.0 では部屋の壁が完全に重なるはずで、
一致セル数は壁の周長 ÷ セル幅（約 104）に近いはずである。
実測は 41 しかない。**間引きで格子に穴が空いているのではないか。**
"""
import os, sys
import numpy as np
sys.path.insert(0, "/home/student/rizu/SNI-SLAM/Registration")
sys.path.insert(0, "/home/student/rizu/SNI-SLAM/Registration/tests")
os.chdir("/home/student/rizu/SNI-SLAM")
from regbim import plan_correlate as pc
from regbim.labels import NAME_TO_ID
from test_plan_candidates import box_room

wall = NAME_TO_ID["wall"]
for n in (26, 60, 140):
    dst_p, dst_l, dst_n = box_room(0, 0, 8, 5, n=n)
    room = box_room(0, 0, 8, 5, n=n)
    cor = box_room(16, 0, 22, 2.2, n=n)
    bp = np.vstack([room[0], cor[0]]); bl = np.concatenate([room[1], cor[1]])
    bn = np.vstack([room[2], cor[2]])
    p = pc.DEFAULTS
    sp, sl, sn = pc._subsample(bp, bl, bn, p["max_points"], p["subsample_seed"])
    dp, dl, dn = pc._subsample(dst_p, dst_l, dst_n, p["max_points"], p["subsample_seed"])
    dm = dl == wall; wm = sl == wall
    dx, dy = pc.wall_groups(dp[dm], dn[dm])
    best = None
    for s in np.exp(np.linspace(np.log(0.6), np.log(1.6), 41)):
        sx, sy = pc.wall_groups(sp[wm] * s, sn[wm])
        hs = pc.horizontal_candidates(sx, sy, dx, dy, p["cell_m"], 1, p["peak_sep_m"])
        if hs and (best is None or hs[0]["score"] > best[1]):
            best = (s, hs[0]["score"])
    ncell_d = len(np.unique(np.floor(np.vstack([dx, dy]) / p["cell_m"]).astype(np.int64), axis=0))
    print("壁の分割数 n=%-4d  総点数 %6d → 間引き後 %5d / 参照の壁セル %4d / "
          "最良の縮尺 %.4f（相関 %.0f）"
          % (n, len(bp), len(sp), ncell_d, best[0], best[1]))
print("\n**真値は 1.0。分割を細かくして 1.0 に寄るなら、原因は点密度である。**")
