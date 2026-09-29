"""R38 §4 の補足（指示外の記述）— 壁を「法線が X 向き」「法線が Y 向き」の 2 群に分けてヨーを出す。

目的：G1 で置いた SLAM の壁が BIM の軸から回っているとき、
- 両群が同じだけ回っている → 全体の回転（G1 の置き方の回転）
- 両群の回り方が違う         → 地図が直交していない（せん断）
を見分ける材料にする。判定には使わない。手順は r38_map_bend.py と同じ（領域・畳み方・bin・±5° 平均）。
"""
import copy, json, os, sys
import numpy as np, yaml
from scipy.spatial import cKDTree
REPO = os.path.expanduser("~/rizu/SNI-SLAM"); os.chdir(REPO)
sys.path.insert(0, "Registration"); sys.path.insert(0, "Registration/scripts")
from r38_map_bend import TARGETS, ROOM_DIST, g1, r35, yaw_deg, vote
from regbim import io_utils, metrics, preprocess

out = {}
for target in TARGETS:
    scene, cond = target.split("__")
    cfg = copy.deepcopy(yaml.safe_load(open("Registration/configs/realdata/%s.yaml" % target)))
    cfg["source"] = dict(cfg["source"], seed=0)
    src = preprocess.prepare(io_utils.load_source_cloud(cfg), cfg)
    dst = io_utils.load_reference_cloud(cfg)
    wall = src.subset(src.class_mask("wall"))
    RG = metrics.decompose_sim3(g1(scene))[0]
    Rm = metrics.decompose_sim3(r35(target))[0]
    P = metrics.apply_sim3(g1(scene), wall.points)
    nG = wall.normals @ RG.T
    room = cKDTree(dst.points[:, :2]).query(P[:, :2], k=1, workers=-1)[0] <= ROOM_DIST
    fx = np.abs(nG[:, 0]) >= np.abs(nG[:, 1])            # 法線が X 向き（Y–Z 面の壁）
    row = {}
    for rname, rm in (("room", room), ("corr", ~room)):
        for fname, fm in (("xwall", fx), ("ywall", ~fx)):
            a = yaw_deg(nG[rm & fm]); am = yaw_deg((wall.normals @ Rm.T)[rm & fm])
            v, vm = vote(a), vote(am)
            row["%s_%s" % (rname, fname)] = {"n": v[2], "g1_mean": v[1], "method_mean": vm[1]}
    out[target] = row
    print("%-14s " % target + " | ".join("%s: n=%5d G1 %+.2f 手法 %+.2f" % (k, v["n"], v["g1_mean"], v["method_mean"])
                                        for k, v in row.items()), flush=True)
json.dump(out, open("Registration/output/diag/r38_wall_family.json", "w"), indent=1)
