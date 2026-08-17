"""追補3 §3 — SNI-SLAM の意味フィールドを迂回し、2D ラベルをメッシュ頂点へ投影する。

背景
----
SNI-SLAM の意味デコーダは本プロジェクトで一度も正しく動いた実績が無い
（[[2026-07-31_r2_semantic_usage_audit]]：52クラス頭は起動時に破棄され、
Replica は `use_gt_semantic: True` で GT を供給していた）。
実データでは意味フィールドが崩壊し、**door が 58.5%**（入力 2D ラベルでは 2.1%）になった。

一方、2D の ADE20K ラベルは全フレーム分あり、カメラ姿勢も確定している。
**メッシュ頂点を可視フレームへ投影して多数決を取れば、崩壊しうる部品を通さずに
頂点ラベルを決められる。**

出力は既存の `final_mesh_semantic.ply` と同じ形式（頂点色が `src/utils/Mesher.py:288`
`decode_segmap` のパレット）なので、下流の `Registration/regbim/io_utils.py:51`
`_load_slam_mesh()` を**無改造で通る**。既存ファイルは上書きせず別名で出す。

可視性の判定
------------
1. 頂点をカメラ座標へ移し、画像内かつ z > 0 のものだけ残す
2. **depth バッファとの整合**：その画素の実測 depth と頂点の z が ``--depth-tol`` 以内
   （オクルージョンで裏の面が投票するのを防ぐ）
3. **法線の向き**：面法線が視線と逆を向いている（表を向いている）ものだけ
4. 票は ``cos`` （法線と視線のなす角）で重み付けできる（``--weight-by-normal``）

使い方
------
    conda activate sni-slam
    python tools/realdata/project_labels_to_mesh.py \
        --mesh output/RealData/m3_room_a/run1/mesh/final_mesh_semantic.ply \
        --scene-dir data/realdata/m3_room_a \
        --config configs/RealData/m3_room_a.yaml \
        --out output/RealData/m3_room_a/run1/mesh/final_mesh_semantic_projected.ply
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import time
from typing import Dict, List

import cv2
import numpy as np
import open3d as o3d
import yaml

# src/utils/Mesher.py:288 decode_segmap と同一（順序 = クラス index 0..5）
CLASS_NAMES = ["background", "wall", "door", "floor", "window", "ceiling"]
LABEL_COLORS = np.array([
    (128, 128, 128),   # 0 background
    (255,  64,  64),   # 1 wall
    (255, 200,  64),   # 2 door
    (180, 220, 255),   # 3 floor
    ( 64, 200, 255),   # 4 window
    (200, 100, 255),   # 5 ceiling
], dtype=np.uint8)

# seg/semantic_classes.pkl の実測値。生ID -> index（src/utils/datasets.py:97-104 と同じ）
RAW_IDS = [0, 93, 37, 40, 97, 31]


def load_c2w_opencv(traj_path: str) -> np.ndarray:
    """traj.txt（素の OpenCV c2w）を読む。convert_stray.py が反転せずに書いている。"""
    rows = [ln.split() for ln in open(traj_path) if ln.strip()]
    return np.asarray([[float(v) for v in r] for r in rows]).reshape(-1, 4, 4)


def frame_no(path: str) -> int:
    m = re.findall(r"\d+", os.path.basename(path))
    return int(m[0]) if m else -1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mesh", required=True, help="幾何を取るメッシュ")
    ap.add_argument("--scene-dir", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frame-stride", type=int, default=5)
    ap.add_argument("--depth-tol", type=float, default=0.12,
                    help="頂点の z と実測 depth の許容差 [m]（オクルージョン判定）")
    ap.add_argument("--weight-by-normal", action="store_true",
                    help="票を法線と視線のなす角の cos で重み付けする")
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    t0 = time.time()
    with open(args.config) as f:
        cam = yaml.safe_load(f)["cam"]
    fx, fy, cx, cy = cam["fx"], cam["fy"], cam["cx"], cam["cy"]
    H, W = cam["H"], cam["W"]

    mesh = o3d.io.read_triangle_mesh(args.mesh)
    mesh.compute_vertex_normals()
    V = np.asarray(mesh.vertices)
    N = np.asarray(mesh.vertex_normals)
    print("mesh: %d vertices" % len(V))

    c2w = load_c2w_opencv(os.path.join(args.scene_dir, "traj.txt"))
    depth_paths = sorted(glob.glob(os.path.join(args.scene_dir, "depth", "depth_*.png")),
                         key=frame_no)
    sem_paths = sorted(glob.glob(os.path.join(args.scene_dir, "semantic_class",
                                              "semantic_class_*.png")), key=frame_no)
    n = min(len(c2w), len(depth_paths), len(sem_paths))
    idxs = list(range(0, n, args.frame_stride))
    print("using %d of %d frames (stride %d)" % (len(idxs), n, args.frame_stride))

    # 生ID -> index の変換表
    raw_to_idx = np.zeros(256, dtype=np.int16)
    raw_to_idx[:] = -1
    for i, raw in enumerate(RAW_IDS):
        raw_to_idx[raw] = i
    raw_to_idx[12] = 4          # blinds -> window（datasets.py:103 のハック）

    votes = np.zeros((len(V), len(CLASS_NAMES)), dtype=np.float32)
    for k, fi in enumerate(idxs):
        Rw = c2w[fi][:3, :3]
        tw = c2w[fi][:3, 3]
        # 世界 -> カメラ（OpenCV: x右 y下 z前）
        p_cam = (V - tw) @ Rw
        z = p_cam[:, 2]
        m = z > 0.05
        if not m.any():
            continue
        u = (p_cam[:, 0] / z) * fx + cx
        v = (p_cam[:, 1] / z) * fy + cy
        m &= (u >= 0) & (u < W) & (v >= 0) & (v < H)
        if not m.any():
            continue

        # 表を向いている面だけ（法線がカメラ側を向く）
        n_cam = N @ Rw
        facing = (n_cam[:, 2] < 0)
        m &= facing

        ui = u[m].astype(np.int32)
        vi = v[m].astype(np.int32)
        zi = z[m]
        vidx = np.flatnonzero(m)

        d = cv2.imread(depth_paths[fi], cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
        dv = d[vi, ui]
        ok = (dv > 0.05) & (np.abs(dv - zi) < args.depth_tol)   # オクルージョン除去
        if not ok.any():
            continue

        s = cv2.imread(sem_paths[fi], cv2.IMREAD_UNCHANGED)
        lab = raw_to_idx[s[vi[ok], ui[ok]]]
        good = lab >= 0
        if not good.any():
            continue
        target = vidx[ok][good]
        cls = lab[good].astype(np.int64)
        if args.weight_by_normal:
            w = np.abs(n_cam[target, 2])
        else:
            w = np.ones(len(target), dtype=np.float32)
        np.add.at(votes, (target, cls), w)

        if (k + 1) % 50 == 0:
            print("  %d/%d frames (%.0fs)" % (k + 1, len(idxs), time.time() - t0),
                  flush=True)

    voted = votes.sum(axis=1) > 0
    labels = np.zeros(len(V), dtype=np.int64)          # 未投票は background(0)
    labels[voted] = votes[voted].argmax(axis=1)
    mesh.vertex_colors = o3d.utility.Vector3dVector(
        LABEL_COLORS[labels].astype(np.float64) / 255.0)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    o3d.io.write_triangle_mesh(args.out, mesh)

    frac = {CLASS_NAMES[i]: round(float((labels == i).mean()), 5)
            for i in range(len(CLASS_NAMES))}
    # 投票された頂点だけのクラス比。入力2Dラベルの画素比と比較すべきはこちら
    # （未投票頂点は background 扱いにしているので、全体比は background が膨らむ）。
    lv = labels[voted]
    frac_voted = {CLASS_NAMES[i]: round(float((lv == i).mean()), 5)
                  for i in range(len(CLASS_NAMES))} if voted.any() else {}
    rep = {
        "mesh_in": args.mesh, "mesh_out": args.out, "scene_dir": args.scene_dir,
        "n_vertices": int(len(V)), "n_frames_used": len(idxs),
        "frame_stride": args.frame_stride, "depth_tol_m": args.depth_tol,
        "weight_by_normal": bool(args.weight_by_normal),
        # 「どの depth 観測からも depth_tol 以内に無い頂点」の割合は、
        # 再構成が観測にどれだけ支持されているかの直接の指標になる
        "vertices_with_votes_frac": round(float(voted.mean()), 5),
        "class_fraction_all": frac,
        "class_fraction_voted_only": frac_voted,
        "elapsed_s": round(time.time() - t0, 1),
    }
    out_json = args.report or (os.path.splitext(args.out)[0] + "_report.json")
    with open(out_json, "w") as f:
        json.dump(rep, f, indent=2, ensure_ascii=False)

    print("\n投票された頂点の割合: %.4f  （残りは depth 観測に支持されていない）"
          % rep["vertices_with_votes_frac"])
    print("クラス比（全頂点）    : %s" % frac)
    print("クラス比（投票済のみ）: %s" % frac_voted)
    print("wrote %s\nwrote %s  (%.0fs)" % (args.out, out_json, rep["elapsed_s"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
