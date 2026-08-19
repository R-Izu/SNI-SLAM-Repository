"""追補4 §5 — Open3D の TSDF 統合で融合し、SNI-SLAM のマッピングと比較する。

なぜ必要か
----------
フォールバック D の構成では、SNI-SLAM に残っている役割は**深度の融合だけ**である
（姿勢＝ARKit VIO、深度＝iPad LiDAR、意味＝ADE20K＋自前投影）。
そして E-7 で「マップが破綻」を撤回した結果、**その融合が良いのか悪いのかは分かっていない**。

同じ姿勢・同じ深度で古典的な TSDF 統合を行い、同じ指標で比較すれば切り分けられる。

出力は SNI-SLAM のメッシュと同じ扱いができるよう三角メッシュの PLY。
意味は付けない（`project_labels_to_mesh.py` で後から付ける。SNI-SLAM 側と同じ経路）。

使い方
------
    conda activate sni-slam
    python tools/realdata/tsdf_fusion.py \
        --scene-dir data/realdata/m3_room_a \
        --config configs/RealData/_D/m3_room_a.yaml \
        --out output/RealData/_TSDF/m3_room_a/mesh/final_mesh_semantic.ply
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time
from typing import List

import cv2
import numpy as np
import open3d as o3d

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, REPO)


def frame_no(p: str) -> int:
    m = re.findall(r"\d+", os.path.basename(p))
    return int(m[0]) if m else -1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene-dir", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--voxel", type=float, default=0.04,
                    help="TSDF のボクセル長 [m]。SNI-SLAM 側の meshing.resolution と揃える")
    ap.add_argument("--sdf-trunc", type=float, default=0.12,
                    help="切り捨て距離。慣例的にボクセル長の 3 倍")
    ap.add_argument("--frame-stride", type=int, default=2)
    ap.add_argument("--max-depth", type=float, default=5.0,
                    help="採用する depth の上限。変換時と同じ 5 m")
    ap.add_argument("--poses", default="traj", choices=["traj", "ckpt"],
                    help="traj: traj.txt の ARKit 姿勢（D と同条件）")
    ap.add_argument("--ckpt-run", default=None)
    args = ap.parse_args()

    os.chdir(REPO)
    from src import config as sni_config
    cam = sni_config.load_config(args.config, "configs/SNI-SLAM.yaml")["cam"]
    fx, fy, cx, cy = cam["fx"], cam["fy"], cam["cx"], cam["cy"]
    H, W = cam["H"], cam["W"]

    if args.poses == "traj":
        rows = [ln.split() for ln in open(os.path.join(args.scene_dir, "traj.txt")) if ln.strip()]
        c2w = np.asarray([[float(v) for v in r] for r in rows]).reshape(-1, 4, 4)
    else:
        import torch
        ck_dir = os.path.join(args.ckpt_run, "ckpts")
        tars = sorted(f for f in os.listdir(ck_dir) if f.endswith(".tar"))
        ck = torch.load(os.path.join(ck_dir, tars[-1]), map_location="cpu")
        c2w = ck["estimate_c2w_list"].numpy()

    rgbs = sorted(glob.glob(os.path.join(args.scene_dir, "rgb", "rgb_*.png")), key=frame_no)
    deps = sorted(glob.glob(os.path.join(args.scene_dir, "depth", "depth_*.png")), key=frame_no)
    n = min(len(c2w), len(rgbs), len(deps))
    idxs = list(range(0, n, args.frame_stride))
    print("TSDF: %d of %d frames, voxel %.3f m, trunc %.3f m"
          % (len(idxs), n, args.voxel, args.sdf_trunc))

    # Open3D は 0.12 前後で integration の名前空間が変わった。
    # src/utils/Mesher.py:71-80 と同じく両方に対応する。
    integ = o3d.pipelines.integration if hasattr(o3d, "pipelines") else o3d.integration
    volume = integ.ScalableTSDFVolume(
        voxel_length=args.voxel, sdf_trunc=args.sdf_trunc,
        color_type=integ.TSDFVolumeColorType.RGB8)
    intr = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)

    t0 = time.time()
    for k, i in enumerate(idxs):
        color = cv2.cvtColor(cv2.imread(rgbs[i], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        depth = cv2.imread(deps[i], cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
        depth[depth > args.max_depth] = 0.0
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.ascontiguousarray(color)),
            o3d.geometry.Image(depth),
            depth_scale=1.0, depth_trunc=args.max_depth,
            convert_rgb_to_intensity=False)
        # Open3D は world-to-camera を取る。traj.txt は OpenCV c2w なのでそのまま逆行列。
        volume.integrate(rgbd, intr, np.linalg.inv(c2w[i]))
        if (k + 1) % 200 == 0:
            print("  %d/%d (%.0fs)" % (k + 1, len(idxs), time.time() - t0), flush=True)

    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    o3d.io.write_triangle_mesh(args.out, mesh)
    el = time.time() - t0

    rep = {
        "scene_dir": args.scene_dir, "out": args.out, "poses": args.poses,
        "voxel_m": args.voxel, "sdf_trunc_m": args.sdf_trunc,
        "frame_stride": args.frame_stride, "n_frames_used": len(idxs),
        "max_depth_m": args.max_depth,
        "n_vertices": int(len(mesh.vertices)), "n_triangles": int(len(mesh.triangles)),
        "elapsed_s": round(el, 1),
    }
    with open(os.path.splitext(args.out)[0] + "_tsdf_report.json", "w") as f:
        json.dump(rep, f, indent=2, ensure_ascii=False)
    print("\n頂点 %d / 三角 %d  所要 %.1f 分" % (len(mesh.vertices), len(mesh.triangles), el / 60))
    print("wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
