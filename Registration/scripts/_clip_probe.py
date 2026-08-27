"""クリップ箱そのものを見る。

75% 被覆で参照の壁 x 範囲が 0.017 m になった。「部分被覆だから壁が減った」のか
「クリッパの箱が意図より狭い」のかを、**箱の座標と点の分布を直接見て**決める。
自分の道具の不具合を研究上の発見として報告しないため（追補4 §1）。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import yaml

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "Registration"))
os.chdir(REPO)

from regbim import io_utils                      # noqa: E402
from regbim.clip import floor_area_m2            # noqa: E402
from regbim.labels import CLASS_NAMES            # noqa: E402

scene = sys.argv[1] if len(sys.argv) > 1 else "room_0"
base = yaml.safe_load(open("Registration/configs/sectionC/%s.yaml" % scene))

full = io_utils.load_reference_cloud(base)
p = full.points
print("%s 参照（切らない）: N=%d" % (scene, len(full)))
print("  world XY 範囲  x[%.2f, %.2f] y[%.2f, %.2f] z[%.2f, %.2f]"
      % (p[:, 0].min(), p[:, 0].max(), p[:, 1].min(), p[:, 1].max(),
         p[:, 2].min(), p[:, 2].max()))
print("  床面積 %.1f m2  クラス別 %s"
      % (floor_area_m2(full),
         {CLASS_NAMES[c]: int((full.labels == c).sum()) for c in np.unique(full.labels)}))

# 壁が world のどこにあるか（箱がそこを切っているかを見るため）
w = full.labels == CLASS_NAMES.index("wall")
print("  壁の world XY  x[%.2f, %.2f] y[%.2f, %.2f]"
      % (p[w, 0].min(), p[w, 0].max(), p[w, 1].min(), p[w, 1].max()))
fl = full.labels == CLASS_NAMES.index("floor")
print("  床の world XY  x[%.2f, %.2f] y[%.2f, %.2f]"
      % (p[fl, 0].min(), p[fl, 0].max(), p[fl, 1].min(), p[fl, 1].max()))

for keep in (0.75, 0.50, 0.30):
    cfg = dict(base)
    cfg["reference"] = dict(base["reference"], clip={"keep_frac": keep, "anchor": [-1, -1]})
    c = io_utils.load_reference_cloud(cfg)
    m = c.meta["clip"]
    print("\nkeep=%.2f  箱 x[%.2f, %.2f] y[%.2f, %.2f]  実被覆 %.3f  N %d -> %d"
          % (keep, m["box_min_xy"][0], m["box_max_xy"][0],
             m["box_min_xy"][1], m["box_max_xy"][1],
             m["coverage_achieved"], m["n_points_before"], m["n_points_after"]))
    print("   クラス別 %s"
          % {CLASS_NAMES[int(k)]: v for k, v in m["class_counts_after"].items()})
    q = c.points
    ww = c.labels == CLASS_NAMES.index("wall")
    if ww.sum():
        print("   残った壁の world XY x[%.2f, %.2f] y[%.2f, %.2f]"
              % (q[ww, 0].min(), q[ww, 0].max(), q[ww, 1].min(), q[ww, 1].max()))
