"""小例の点密度が、セル幅 0.25 m に対して足りているかを確かめる。

**前回の密度確認は誤っていた**：n=60/140 では全クラスまとめた間引きが効いて
壁がやせており、密度を上げた効果が打ち消されていた。
**役割ごとの間引きに直した今、density を上げ直して見る。**
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

print("%-6s %10s %12s %12s %12s %10s"
      % ("n", "壁の間隔[m]", "参照の壁セル", "真値のスコア", "勝者のスコア", "勝者の縮尺"))
print("-" * 70)
for n in (26, 40, 60, 100, 160):
    dst_p, dst_l, dst_n = box_room(0, 0, 8, 5, n=n)
    room = box_room(0, 0, 8, 5, n=n); cor = box_room(16, 0, 22, 2.2, n=n)
    bp = np.vstack([room[0], cor[0]]); bl = np.concatenate([room[1], cor[1]])
    bn = np.vstack([room[2], cor[2]])
    sp, sl, sn = pc._subsample_by_role(bp, bl, bn, p["max_points"],
                                       p["subsample_seed"], wall, floor, ceil)
    dp, dl, dn = pc._subsample_by_role(dst_p, dst_l, dst_n, p["max_points"],
                                       p["subsample_seed"], wall, floor, ceil)
    dm, wm = dl == wall, sl == wall
    dx, dy = pc.wall_groups(dp[dm], dn[dm])
    ncell = len(np.unique(np.floor(np.vstack([dx, dy]) / p["cell_m"]).astype(np.int64), axis=0))
    best = None; at_true = None
    for i, s in enumerate(scales):
        sx, sy = pc.wall_groups(sp[wm] * s, sn[wm])
        hs = pc.horizontal_candidates(sx, sy, dx, dy, p["cell_m"], 1, p["peak_sep_m"])
        if not hs: continue
        if best is None or hs[0]["score"] > best[1]: best = (s, hs[0]["score"])
        if i == k: at_true = (s, hs[0]["score"])
    # 長い方の壁（8 m を n 分割）の点間隔
    print("%-6d %10.3f %12d %12.0f %12.0f %10.4f"
          % (n, 8.0 / (n - 1), ncell, at_true[1], best[1], best[0]))
print("\nセル幅 %.2f m。**点間隔がセル幅を超えると占有格子に穴が空く。**" % p["cell_m"])
print("**真値のスコアが勝者に追いつけば、原因は小例の点密度である。**")
