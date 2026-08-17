"""追補3 A-3 — 学習した幾何が入力 depth をどれだけ再現できているかを測る。

目的
----
ATE が悪いとき、それが**トラッキングが悪い**からなのか
**マップが悪くて追う対象が無い**からなのかを切り分ける。
マップ側の収束を、ATE とは独立に測る。

方法
----
入力 depth を逆投影した「観測点群」と、再構成メッシュの頂点を双方向で比較する
（GT メッシュが無い実データでも成立する。`src/tools/eval_recon.py` は GT メッシュ前提）。

    accuracy   : メッシュ頂点 -> 最近傍の観測点 の距離
                 大きい＝**観測に裏付けられない偽の面**が多い
    completion : 観測点 -> 最近傍のメッシュ頂点 の距離
                 大きい＝**見えているのに再構成されていない**

姿勢の選び方（``--poses``）
    est : SLAM が推定した姿勢（ckpt の estimate_c2w_list）。**既定**。
          「マップは自分が信じた姿勢に対してすら整合しているか」を見るので、
          姿勢誤差を混ぜずにマップ単体の質を測れる
    gt  : traj.txt の ARKit 姿勢。姿勢誤差込みの見え方になる

使い方
------
    conda activate sni-slam
    python tools/realdata/eval_map_vs_depth.py \
        --run output/RealData/m3_room_a/run1 \
        --scene-dir data/realdata/m3_room_a \
        --config configs/RealData/m3_room_a.yaml
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from typing import Dict, List

import cv2
import numpy as np
import open3d as o3d
import yaml
from scipy.spatial import cKDTree


def frame_no(p: str) -> int:
    m = re.findall(r"\d+", os.path.basename(p))
    return int(m[0]) if m else -1


def voxel_ds(p: np.ndarray, v: float) -> np.ndarray:
    if len(p) == 0:
        return p
    keys = np.floor(p / v).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return p[np.sort(idx)]


def load_poses(run: str, scene_dir: str, which: str) -> np.ndarray:
    """OpenCV c2w を返す。est は ckpt から、gt は traj.txt から。"""
    if which == "gt":
        rows = [ln.split() for ln in open(os.path.join(scene_dir, "traj.txt")) if ln.strip()]
        return np.asarray([[float(v) for v in r] for r in rows]).reshape(-1, 4, 4)
    import torch
    ck_dir = os.path.join(run, "ckpts")
    tars = sorted(f for f in os.listdir(ck_dir) if f.endswith(".tar"))
    ck = torch.load(os.path.join(ck_dir, tars[-1]), map_location="cpu")
    m = ck["estimate_c2w_list"]
    return (m.numpy() if hasattr(m, "numpy") else np.asarray(m))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="output/RealData/<scene>/runN")
    ap.add_argument("--scene-dir", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--mesh", default=None, help="既定は <run>/mesh/final_mesh_semantic.ply")
    ap.add_argument("--poses", default="est", choices=["est", "gt"])
    ap.add_argument("--frame-stride", type=int, default=20)
    ap.add_argument("--pix-stride", type=int, default=4)
    ap.add_argument("--voxel", type=float, default=0.02)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    # config は inherit_from の連鎖を持つので、リポジトリのローダで解決する
    # （Replica 系は cam を replica.yaml から継承しており、葉だけ読むと足りない）
    sys.path.insert(0, os.path.abspath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..")))
    from src import config as sni_config
    cam = sni_config.load_config(args.config, "configs/SNI-SLAM.yaml")["cam"]
    fx, fy, cx, cy = cam["fx"], cam["fy"], cam["cx"], cam["cy"]

    mesh_path = args.mesh or os.path.join(args.run, "mesh", "final_mesh_semantic.ply")
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    V = voxel_ds(np.asarray(mesh.vertices), args.voxel)
    print("mesh vertices: %d -> %d after %.0f cm voxel" % (
        len(mesh.vertices), len(V), args.voxel * 100))

    c2w = load_poses(args.run, args.scene_dir, args.poses)
    depths = sorted(glob.glob(os.path.join(args.scene_dir, "depth", "depth_*.png")),
                    key=frame_no)
    n = min(len(c2w), len(depths))
    idxs = list(range(0, n, args.frame_stride))
    print("poses=%s, using %d of %d frames" % (args.poses, len(idxs), n))

    obs: List[np.ndarray] = []
    for i in idxs:
        d = cv2.imread(depths[i], cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
        vv, uu = np.mgrid[0:d.shape[0]:args.pix_stride, 0:d.shape[1]:args.pix_stride]
        z = d[::args.pix_stride, ::args.pix_stride]
        m = z > 0.05
        if not m.any():
            continue
        u, v, z = uu[m].astype(np.float64), vv[m].astype(np.float64), z[m].astype(np.float64)
        p_cam = np.stack([(u - cx) / fx * z, (v - cy) / fy * z, z], axis=1)
        obs.append(p_cam @ c2w[i][:3, :3].T + c2w[i][:3, 3])
    O = voxel_ds(np.concatenate(obs, axis=0), args.voxel)
    print("observed points: %d after voxel" % len(O))

    # accuracy: メッシュ頂点 -> 観測点
    d_acc, _ = cKDTree(O).query(V, k=1)
    # completion: 観測点 -> メッシュ頂点
    d_comp, _ = cKDTree(V).query(O, k=1)

    def stats(d: np.ndarray, name: str) -> Dict[str, float]:
        return {
            "%s_median_m" % name: round(float(np.median(d)), 4),
            "%s_mean_m" % name: round(float(d.mean()), 4),
            "%s_p90_m" % name: round(float(np.percentile(d, 90)), 4),
            "%s_frac_within_5cm" % name: round(float((d < 0.05).mean()), 4),
            "%s_frac_within_10cm" % name: round(float((d < 0.10).mean()), 4),
            "%s_frac_within_20cm" % name: round(float((d < 0.20).mean()), 4),
        }

    rep: Dict[str, object] = {
        "run": args.run, "mesh": mesh_path, "poses": args.poses,
        "n_mesh_vertices_ds": int(len(V)), "n_observed_points_ds": int(len(O)),
        "voxel_m": args.voxel, "frame_stride": args.frame_stride,
    }
    rep.update(stats(d_acc, "accuracy"))
    rep.update(stats(d_comp, "completion"))

    out = args.out or os.path.join(args.run, "map_vs_depth_%s.json" % args.poses)
    with open(out, "w") as f:
        json.dump(rep, f, indent=2, ensure_ascii=False)

    print("\naccuracy   (メッシュ頂点 -> 観測点): 中央値 %.3f m / 5cm以内 %.1f%% / 10cm以内 %.1f%%"
          % (rep["accuracy_median_m"], 100 * rep["accuracy_frac_within_5cm"],
             100 * rep["accuracy_frac_within_10cm"]))
    print("completion (観測点 -> メッシュ頂点): 中央値 %.3f m / 5cm以内 %.1f%% / 10cm以内 %.1f%%"
          % (rep["completion_median_m"], 100 * rep["completion_frac_within_5cm"],
             100 * rep["completion_frac_within_10cm"]))
    print("\naccuracy が大きい = 観測に裏付けられない偽の面が多い")
    print("completion が大きい = 見えているのに再構成されていない")
    print("wrote %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
