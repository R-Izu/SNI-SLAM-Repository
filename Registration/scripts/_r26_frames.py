"""R26 §1 の前段：**そもそも座標系はいくつあるのか**を数値で確かめる。

**独立 GT を作る前に、これを確かめないと何と何を比べているか分からない。**

確かめること
------------
1. `traj.txt`（Replica の GT カメラ姿勢）の世界座標系と、
   SLAM が mesh を作った世界座標系は**同じか**
   → `src/utils/datasets.py:198` の `c2w[:3,1] *= -1` はカメラ**局所**軸の反転であり、
     並進列 `c2w[:3,3]` に触らない。**世界座標系は変わらないはず**。確かめる
2. `traj.txt` の世界座標系と、参照点群（habitat メッシュ）の座標系は**同じか**
   → カメラ位置が参照の外接箱に入っていれば同じ。**入っていなければ別**

**何も登録しない。姿勢と点群の座標を並べるだけ。**

    python Registration/scripts/_r26_frames.py room_0 \\
        output/Replica/room0_official/260310_test4/ckpts/01999.tar
"""

import os
import sys

import numpy as np
import torch
import yaml

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "Registration"))
os.chdir(REPO)

import open3d as o3d                                   # noqa: E402
from regbim import io_utils                            # noqa: E402

SCENE = sys.argv[1] if len(sys.argv) > 1 else "room_0"
CKPT = sys.argv[2] if len(sys.argv) > 2 else \
    "output/Replica/room0_official/260310_test4/ckpts/01999.tar"
SEQ = SCENE + "_official"

traj = np.loadtxt("data/replica/%s/traj.txt" % SEQ).reshape(-1, 4, 4)
cam = traj[:, :3, 3]
print("traj.txt: %d 姿勢" % len(traj))
print("  カメラ位置の範囲 x [%.2f, %.2f]  y [%.2f, %.2f]  z [%.2f, %.2f]"
      % (cam[:, 0].min(), cam[:, 0].max(), cam[:, 1].min(), cam[:, 1].max(),
         cam[:, 2].min(), cam[:, 2].max()))

cfg = yaml.safe_load(open("Registration/configs/sectionC/%s.yaml" % SCENE))
v = io_utils.load_reference_cloud(cfg).points
print("\n参照点群（habitat メッシュを既存ローダで読んだもの）: %d 点" % len(v))
print("  範囲 x [%.2f, %.2f]  y [%.2f, %.2f]  z [%.2f, %.2f]"
      % (v[:, 0].min(), v[:, 0].max(), v[:, 1].min(), v[:, 1].max(),
         v[:, 2].min(), v[:, 2].max()))

inside = ((cam >= v.min(0)) & (cam <= v.max(0))).all(axis=1)
print("\n**カメラ位置が参照の外接箱に入っている割合: %d / %d (%.1f%%)**"
      % (inside.sum(), len(cam), 100.0 * inside.mean()))
print("  → 100%% なら traj.txt と参照は**同じ座標系**")
print("  → 低いなら**別の座標系**であり、その間の変換を別途決める必要がある")

ck = torch.load(CKPT, map_location="cpu")
print("\nckpt のキー: %s" % sorted(ck.keys()))
for k in ("estimate_c2w_list", "gt_c2w_list"):
    if k in ck:
        a = np.asarray(ck[k])
        p = a[:, :3, 3]
        print("  %-18s shape %s   位置の範囲 x [%.2f, %.2f] y [%.2f, %.2f] z [%.2f, %.2f]"
              % (k, a.shape, p[:, 0].min(), p[:, 0].max(), p[:, 1].min(),
                 p[:, 1].max(), p[:, 2].min(), p[:, 2].max()))

src = o3d.io.read_triangle_mesh(cfg["source"]["mesh_path"])
sv = np.asarray(src.vertices)
print("\nSLAM メッシュ: %d 頂点   範囲 x [%.2f, %.2f] y [%.2f, %.2f] z [%.2f, %.2f]"
      % (len(sv), sv[:, 0].min(), sv[:, 0].max(), sv[:, 1].min(), sv[:, 1].max(),
         sv[:, 2].min(), sv[:, 2].max()))

t_gt = "Registration/output/sectionC/%s/T_gt.json" % SCENE
for cand in (t_gt, cfg.get("eval", {}).get("t_gt_path", "")):
    if cand and os.path.exists(cand):
        print("\n凍結された旧 T_gt: %s" % cand)
        break
else:
    print("\n凍結された旧 T_gt が既定の場所に見つからない。探すこと")
