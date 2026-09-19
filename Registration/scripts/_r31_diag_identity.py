"""恒等変換の場合だけ候補が遠い理由を切り分ける。

**縮尺の格子が 1.0 を含んでいないのか、別のピークを選んでいるのか。**
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

scales = np.exp(np.linspace(np.log(0.3), np.log(3.0), 59))
k = int(np.argmin(np.abs(scales - 1.0)))
print("縮尺格子で 1.0 に最も近い値: %.6f（%.3f%% ずれ）"
      % (scales[k], 100 * (scales[k] - 1.0)))
print("格子の間隔（対数）: %.5f → 隣どうしで %.2f%% 違う"
      % (np.log(scales[1]) - np.log(scales[0]),
         100 * (scales[1] / scales[0] - 1)))

dst_p, dst_l, dst_n = box_room(0, 0, 8, 5)
room = box_room(0, 0, 8, 5)
cor = box_room(16, 0, 22, 2.2)
base_p = np.vstack([room[0], cor[0]])
base_l = np.concatenate([room[1], cor[1]])
base_n = np.vstack([room[2], cor[2]])

cands, diag = pc.generate_candidates(base_p, base_l, base_n,
                                     dst_p, dst_l, dst_n, [np.eye(3)], NAME_TO_ID)
print("\n候補 %d 個（真値は s=1.0, t=(0,0,0)）" % len(cands))
print("%-4s %9s %22s %9s %9s %9s"
      % ("id", "縮尺", "水平移動", "dz", "水平相関", "誤差[m]"))
for c in cands:
    moved = base_p @ c["T"][:3, :3].T + c["T"][:3, 3]
    err = float(np.sqrt(((moved - base_p) ** 2).sum(axis=1).mean()))
    print("%-4d %9.4f %22s %9.3f %9.0f %9.3f"
          % (c["cand_id"], c["scale"],
             np.round(c["shift_xy"], 2).tolist(), c["dz"],
             c["score_h"], err))

print("\n**source の広がり**: x [%.1f, %.1f]  参照: x [%.1f, %.1f]"
      % (base_p[:, 0].min(), base_p[:, 0].max(),
         dst_p[:, 0].min(), dst_p[:, 0].max()))
print("縮尺が %.3f%% ずれると、x=%.0f m の点は %.2f m 動く"
      % (100 * abs(scales[k] - 1.0), base_p[:, 0].max(),
         abs(scales[k] - 1.0) * base_p[:, 0].max()))
