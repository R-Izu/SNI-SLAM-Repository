"""R61 §3-4 の前の確認：前処理（`preprocess.prepare`）の後の点が、P_xy・P_yaw で摂動なしの点（P をかけて戻したもの）と同じか。m3_cor_a__E2・k 0 の 1 試行。位置合わせは回さない。回す前に commit"""
import sys, os, numpy as np
sys.path.insert(0, os.path.expanduser("~/rizu/SNI-SLAM/Registration/scripts/r61"))
import r61_run as M
from regbim import metrics, preprocess
import open3d as o3d, inspect
print("open3d", o3d.__version__)
t = M.RR.REAL[0]; src, dst, _ = M.RR.clouds("real", t); cfg = M.RR.config("real", t, "full")
P0 = M.RR.perturbations("real", t)[0]; c, _ = M.components(P0)
a = preprocess.prepare(src, cfg)
for name in ("xy", "yaw"):
    P = c[name]; b = preprocess.prepare(metrics.transform_cloud(src, P), cfg)
    back = metrics.transform_cloud(b, metrics.invert_sim3(P))
    A = a.points[np.lexsort(a.points.T)]; B = back.points[np.lexsort(back.points.T)]
    same_n = len(A) == len(B)
    print(name, "n", len(a.points), len(b.points), "max|diff| after sort" , float(np.abs(A - B).max()) if same_n else None)
