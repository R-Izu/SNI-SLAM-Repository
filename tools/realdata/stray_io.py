"""Stray Scanner (iPad Pro / LiDAR) 出力の読み出しユーティリティ。

Stray Scanner のスキャンディレクトリは以下の構成をとる::

    <scan_id>/
      rgb.mp4            1920x1440 / 30fps / H.264
      depth/*.png        256x192 uint16 (mm)
      confidence/*.png   256x192 uint8  (0/1/2)
      odometry.csv       timestamp, frame, x,y,z, qx,qy,qz,qw, fx,fy,cx,cy, ...
      imu.csv, camera_matrix.csv

姿勢規約について:
    Report 2026-08-10_r2_realdata_slam_pipeline.md §2-C の実測により、
    ``odometry.csv`` の (位置, クォータニオン) は **そのまま OpenCV 規約
    (x右・y下・z前) の camera-to-world 行列** として成立する（追加の軸反転は不要）。
    世界座標系は ARKit 準拠で **+Y が鉛直上向き**。
    本モジュールは既定でこの解釈を用いるが、``convention="opengl"`` を渡すと
    y,z を反転した解釈（対立仮説）も構成でき、両者を数値比較できる。
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# 定数
# ---------------------------------------------------------------------------

RGB_W, RGB_H = 1920, 1440       # Stray の rgb.mp4 解像度
DEPTH_W, DEPTH_H = 256, 192     # Stray の depth/confidence 解像度
DEPTH_SCALE = 1000.0            # uint16 (mm) -> m
FPS = 30.0


# ---------------------------------------------------------------------------
# odometry
# ---------------------------------------------------------------------------

def read_odometry(scan_dir: str) -> Dict[str, np.ndarray]:
    """``odometry.csv`` を読み、姿勢と per-frame 内部パラメータを返す。

    Returns
    -------
    dict with keys:
        timestamp (N,), frame (N,) int, pos (N,3), quat (N,4) [qx,qy,qz,qw],
        intr (N,4) [fx,fy,cx,cy]  ※ RGB 解像度 (1920x1440) 基準
    """
    path = os.path.join(scan_dir, "odometry.csv")
    rows: List[List[float]] = []
    with open(path, "r") as f:
        header = f.readline()  # noqa: F841  ヘッダ行は読み捨て
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            # distortion_center_* は空欄のことがあるので先頭 13 列のみ使う
            rows.append([float(p) for p in parts[:13]])
    a = np.asarray(rows, dtype=np.float64)
    return {
        "timestamp": a[:, 0],
        "frame": a[:, 1].astype(np.int64),
        "pos": a[:, 2:5],
        "quat": a[:, 5:9],          # qx, qy, qz, qw
        "intr": a[:, 9:13],         # fx, fy, cx, cy (RGB 解像度基準)
    }


def quat_to_rotmat(quat: np.ndarray) -> np.ndarray:
    """[qx,qy,qz,qw] (N,4) -> 回転行列 (N,3,3)。"""
    q = np.asarray(quat, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((len(q), 3, 3), dtype=np.float64)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - z * w)
    R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w)
    R[:, 2, 1] = 2 * (y * z + x * w)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def build_c2w(odo: Dict[str, np.ndarray], convention: str = "opencv") -> np.ndarray:
    """odometry から camera-to-world 行列 (N,4,4) を作る。

    convention="opencv" : そのまま採用（Report §2-C の採用解釈）
    convention="opengl" : カメラ軸 y,z を反転した対立解釈（比較用）
    """
    R = quat_to_rotmat(odo["quat"])
    t = odo["pos"]
    if convention == "opengl":
        flip = np.diag([1.0, -1.0, -1.0])
        R = R @ flip
    elif convention != "opencv":
        raise ValueError("convention must be 'opencv' or 'opengl', got %r" % convention)
    n = len(R)
    c2w = np.tile(np.eye(4), (n, 1, 1))
    c2w[:, :3, :3] = R
    c2w[:, :3, 3] = t
    return c2w


# ---------------------------------------------------------------------------
# depth / confidence
# ---------------------------------------------------------------------------

def list_depth_files(scan_dir: str) -> List[str]:
    d = os.path.join(scan_dir, "depth")
    return sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".png"))


def list_confidence_files(scan_dir: str) -> List[str]:
    d = os.path.join(scan_dir, "confidence")
    if not os.path.isdir(d):
        return []
    return sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".png"))


def read_depth_m(path: str) -> np.ndarray:
    """depth PNG (uint16 mm) -> float32 (m)。0 は無効値としてそのまま 0。"""
    raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise IOError("cannot read depth png: %s" % path)
    return raw.astype(np.float32) / DEPTH_SCALE


def read_confidence(path: str) -> np.ndarray:
    raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise IOError("cannot read confidence png: %s" % path)
    return raw


def depth_intrinsics(intr_rgb: np.ndarray) -> Tuple[float, float, float, float]:
    """RGB 解像度基準の (fx,fy,cx,cy) を depth 解像度 (256x192) 基準へ換算。"""
    sx = DEPTH_W / float(RGB_W)
    sy = DEPTH_H / float(RGB_H)
    fx, fy, cx, cy = intr_rgb
    return fx * sx, fy * sy, cx * sx, cy * sy


# ---------------------------------------------------------------------------
# 逆投影
# ---------------------------------------------------------------------------

def backproject_frame(
    depth_m: np.ndarray,
    conf: Optional[np.ndarray],
    intr_rgb: np.ndarray,
    c2w: np.ndarray,
    conf_min: int = 1,
    pix_stride: int = 1,
    max_depth: float = 8.0,
) -> np.ndarray:
    """1 フレームを世界座標へ逆投影して (M,3) を返す。

    カメラ規約は OpenCV (x右・y下・z前)。``c2w`` は 4x4。
    """
    fx, fy, cx, cy = depth_intrinsics(intr_rgb)
    h, w = depth_m.shape
    vs, us = np.mgrid[0:h:pix_stride, 0:w:pix_stride]
    z = depth_m[::pix_stride, ::pix_stride]
    m = (z > 0.05) & (z < max_depth)
    if conf is not None and conf_min > 0:
        m &= conf[::pix_stride, ::pix_stride] >= conf_min
    if not m.any():
        return np.zeros((0, 3), dtype=np.float32)
    z = z[m].astype(np.float64)
    u = us[m].astype(np.float64)
    v = vs[m].astype(np.float64)
    p_cam = np.stack([(u - cx) / fx * z, (v - cy) / fy * z, z], axis=1)
    p_w = p_cam @ c2w[:3, :3].T + c2w[:3, 3]
    return p_w.astype(np.float32)


# ---------------------------------------------------------------------------
# RGB
# ---------------------------------------------------------------------------

def rgb_frame_count(scan_dir: str) -> int:
    """rgb.mp4 のフレーム数（コンテナのメタデータ由来。デコード実数ではない）。"""
    cap = cv2.VideoCapture(os.path.join(scan_dir, "rgb.mp4"))
    try:
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()


def extract_rgb_frames(scan_dir: str, indices: List[int]) -> Dict[int, np.ndarray]:
    """指定インデックスの RGB フレームを BGR 配列で取り出す（順次デコード）。"""
    want = sorted(set(int(i) for i in indices))
    out: Dict[int, np.ndarray] = {}
    if not want:
        return out
    cap = cv2.VideoCapture(os.path.join(scan_dir, "rgb.mp4"))
    try:
        idx, ptr = 0, 0
        while ptr < len(want):
            ok, frame = cap.read()
            if not ok:
                break
            if idx == want[ptr]:
                out[idx] = frame
                ptr += 1
            idx += 1
    finally:
        cap.release()
    return out
