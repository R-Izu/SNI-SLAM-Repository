"""小例の壁が、セル境界にぴったり乗っているせいではないか。

x=±4.0, y=±2.5 はどれも 0.25 の整数倍である。
**境界上にある面は、わずかに縮めるだけでセル添字が 1 つ飛ぶ。**
向かい合う2面で飛び方が違えば、整数セルの平行移動では合わせられない。

**実データの壁が 0.25 m の整数倍に乗る理由はない。**
そこで部屋全体を非整数倍だけずらして、同じことが起きるかを見る。
"""
import os, sys
import numpy as np
sys.path.insert(0, "/home/student/rizu/SNI-SLAM/Registration")
sys.path.insert(0, "/home/student/rizu/SNI-SLAM/Registration/tests")
os.chdir("/home/student/rizu/SNI-SLAM")
from regbim import plan_correlate as pc
from regbim.labels import NAME_TO_ID
from test_plan_candidates import box_room

p = pc.DEFAULTS
wall, floor, ceil = (NAME_TO_ID["wall"], NAME_TO_ID["floor"], NAME_TO_ID["ceiling"])
scales = np.exp(np.linspace(np.log(p["scale_lo"]), np.log(p["scale_hi"]), p["scale_n"]))
k = int(np.argmin(abs(scales - 1.0)))
print("真値に最も近い格子点 s=%.6f\n" % scales[k])

print("%-14s %12s %12s %12s %s"
      % ("部屋のずらし", "真値スコア", "勝者スコア", "勝者の縮尺", "真値が勝つか"))
print("-" * 68)
for off in (0.0, 0.07, 0.11, 0.13, 0.17):
    d = box_room(off, off, 8, 5, n=60)
    room = box_room(off, off, 8, 5, n=60)
    cor = box_room(16 + off, off, 22, 2.2, n=60)
    bp = np.vstack([room[0], cor[0]]); bl = np.concatenate([room[1], cor[1]])
    bn = np.vstack([room[2], cor[2]])
    sp, sl, sn = pc._subsample_by_role(bp, bl, bn, p["max_points"],
                                       p["subsample_seed"], wall, floor, ceil)
    dp, dl, dn = pc._subsample_by_role(d[0], d[1], d[2], p["max_points"],
                                       p["subsample_seed"], wall, floor, ceil)
    dm, wm = dl == wall, sl == wall
    dx, dy = pc.wall_groups(dp[dm], dn[dm])
    best = None; at_true = None
    for i, s in enumerate(scales):
        sx, sy = pc.wall_groups(sp[wm] * s, sn[wm])
        hs = pc.horizontal_candidates(sx, sy, dx, dy, p["cell_m"], 1, p["peak_sep_m"])
        if not hs: continue
        if best is None or hs[0]["score"] > best[1]: best = (s, hs[0]["score"])
        if i == k: at_true = (s, hs[0]["score"])
    win = abs(best[0] - 1.0) < 0.02
    print("%-14.2f %12.0f %12.0f %12.4f %s"
          % (off, at_true[1], best[1], best[0], "**はい**" if win else "いいえ"))

print("\n**ずらすと真値が勝つなら、原因は小例がセル境界に乗っていたことである。**")
