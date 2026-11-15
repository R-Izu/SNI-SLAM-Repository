"""R57 §3-3 の道具：座標の確認のための密な IFC の点群（bim-ifc の環境）。**回す前に commit する。**
参照の作り方（20 万点）では点の間隔が粗いので、同じ `build_ifc_cloud`・同じ class_map で 5,000,000 点にしたものを、距離を測る道具として別に作る。
出力：`~/rizu/conslam/out/conslam_ifc_dense5m.npz`
"""
import json, os, sys
import numpy as np

REPO = os.path.expanduser("~/rizu/SNI-SLAM")
sys.path.insert(0, os.path.join(REPO, "Registration", "scripts"))
import ifc_export as X                                         # noqa: E402

IFC = "/mnt/d/rizu/SNI-SLAM/data/ConSLAM/m1743877/ConSLAM_Building_00_GroundTruth_V9_wall_segment.ifc"
OUT = os.path.expanduser("~/rizu/conslam/out")
cm = json.load(open(os.path.join(OUT, "class_map.json")))
res = X.build_ifc_cloud(IFC, cm, 5000000, spaces=None, storeys=None, seed=0, keep_classes=["wall", "floor", "ceiling", "door", "window"])
np.savez_compressed(os.path.join(OUT, "conslam_ifc_dense5m.npz"), points=np.asarray(res["points"], dtype=np.float32), labels=np.asarray(res["labels"]))
print(len(res["points"]))
