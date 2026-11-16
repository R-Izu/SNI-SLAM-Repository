"""R62 §3-3 の材料：密な IFC の点群を法線つきで作り直す（bim-ifc の環境）。**回す前に commit する。**
R57 の `conslam_ifc_dense5m.npz` と同じ `build_ifc_cloud`・同じ class_map・5,000,000 点・seed 0。R57 版は法線を保存していなかったので、法線も書く。
出力：`~/rizu/conslam/out/conslam_ifc_dense5m_n.npz`（points・labels・normals）。点と クラスが R57 版と同じかも確かめる
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
p = np.asarray(res["points"], dtype=np.float32); l = np.asarray(res["labels"]); n = np.asarray(res["normals"], dtype=np.float32)
np.savez_compressed(os.path.join(OUT, "conslam_ifc_dense5m_n.npz"), points=p, labels=l, normals=n)
old = np.load(os.path.join(OUT, "conslam_ifc_dense5m.npz"))
print(len(p), "same_points_as_r57", bool(np.array_equal(p, old["points"])), "same_labels", bool(np.array_equal(l, old["labels"])))
