import os, sys
import numpy as np
sys.path.insert(0, "/home/student/rizu/SNI-SLAM/Registration")
sys.path.insert(0, "/home/student/rizu/SNI-SLAM/Registration/tests")
os.chdir("/home/student/rizu/SNI-SLAM")
from regbim import plan_correlate as pc
from regbim.labels import NAME_TO_ID
from test_plan_candidates import box_room

wall = NAME_TO_ID["wall"]
dst_p, dst_l, dst_n = box_room(0, 0, 8, 5, n=60)
room = box_room(0, 0, 8, 5, n=60)
cor = box_room(16, 0, 22, 2.2, n=60)
bp = np.vstack([room[0], cor[0]]); bl = np.concatenate([room[1], cor[1]])
bn = np.vstack([room[2], cor[2]])
cell = 0.25

# 間引き無しで、真の対応（s=1, shift 0）の一致数を直に数える
wm = bl == wall; dm = dst_l == wall
sx, sy = pc.wall_groups(bp[wm], bn[wm])
dx, dy = pc.wall_groups(dst_p[dm], dst_n[dm])
print("source 壁 x群 %d 点 / y群 %d 点" % (len(sx), len(sy)))
print("参照   壁 x群 %d 点 / y群 %d 点" % (len(dx), len(dy)))

def cells(a):
    return set(map(tuple, np.floor(a / cell).astype(np.int64)))

for nm, s_, d_ in (("x群", sx, dx), ("y群", sy, dy)):
    cs, cd = cells(s_), cells(d_)
    print("  %s: source %d セル / 参照 %d セル / **同じ位置で一致 %d セル**"
          % (nm, len(cs), len(cd), len(cs & cd)))

tot = len(cells(sx) & cells(dx)) + len(cells(sy) & cells(dy))
print("\n真の対応（s=1, shift 0）での一致セル数 = %d" % tot)

hs = pc.horizontal_candidates(sx, sy, dx, dy, cell, 5, 0.5)
print("\n相関の上位ピーク（間引き無し）:")
for h in hs:
    print("   移動 %-18s 相関 %.0f" % (np.round(h["shift_xy"], 2).tolist(), h["score"]))
