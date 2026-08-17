"""traj.txt が SLAM の規約と整合しているかを、逆投影して実測で確かめる。

`src/utils/datasets.py:199-200`（col 1,2 の反転）と `src/common.py:90`
（`dirs = [(i-cx)/fx, -(j-cy)/fy, -1]`）を**そのまま再現**して depth を逆投影し、

  - 最下面（下位3%）に平面を当てた法線が鉛直か
  - 床から天井までの高さが S0 の実測値（約 2.58 m）と合うか

を見る。姿勢規約を間違えると SLAM は「動くが再構成が壊れる」という形で失敗し、
完走したかどうかでは検出できないため、この確認を変換の受入条件に入れる。
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List

import cv2
import numpy as np
import yaml

NORMAL_MAX_DEG = 3.0        # 最下面の法線と鉛直のなす角の上限
CEILING_RANGE_M = (2.2, 3.6)


def read_cam(cfg_path: str) -> Dict[str, float]:
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    return cfg["cam"]


def check_scene(scene_dir: str, cfg_path: str, n_frames: int = 12,
                pix_stride: int = 8) -> Dict[str, object]:
    cam = read_cam(cfg_path)
    fx, fy, cx, cy = cam["fx"], cam["fy"], cam["cx"], cam["cy"]
    rows = [ln.split() for ln in open(os.path.join(scene_dir, "traj.txt")) if ln.strip()]
    M = np.asarray([[float(v) for v in r] for r in rows]).reshape(-1, 4, 4)

    depths = sorted(glob.glob(os.path.join(scene_dir, "depth", "depth_*.png")),
                    key=lambda p: int(os.path.basename(p)[6:-4]))
    n = min(len(M), len(depths))
    idxs = np.linspace(int(n * 0.05), int(n * 0.95), n_frames).astype(int)

    pts: List[np.ndarray] = []
    for k in idxs:
        c2w = M[k].copy()
        c2w[:3, 1] *= -1          # datasets.py:199
        c2w[:3, 2] *= -1          # datasets.py:200
        d = cv2.imread(depths[k], cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
        j, i = np.mgrid[0:d.shape[0]:pix_stride, 0:d.shape[1]:pix_stride]
        z = d[::pix_stride, ::pix_stride]
        m = z > 0.05
        if not m.any():
            continue
        i, j, z = i[m], j[m], z[m]
        dirs = np.stack([(i - cx) / fx, -(j - cy) / fy, -np.ones_like(i)], -1)  # common.py:90
        rays = dirs @ c2w[:3, :3].T
        rays /= np.linalg.norm(rays, axis=1, keepdims=True)
        pts.append(c2w[:3, 3] + rays * z[:, None])
    if not pts:
        return {"error": "no points"}
    P = np.concatenate(pts)

    low = P[P[:, 1] < np.percentile(P[:, 1], 3)]
    c = low - low.mean(0)
    nrm = np.linalg.svd(c, full_matrices=False)[2][-1]
    ang = float(np.degrees(np.arccos(np.clip(abs(nrm[1]), -1, 1))))

    lo, hi = np.percentile(P[:, 1], [0.5, 99.5])
    return {
        "floor_normal_vs_up_deg": round(ang, 3),
        "y_low": round(float(lo), 3),
        "y_high": round(float(hi), 3),
        "y_span_m": round(float(hi - lo), 3),
        "normal_ok": bool(ang <= NORMAL_MAX_DEG),
        "span_ok": bool(CEILING_RANGE_M[0] <= hi - lo <= CEILING_RANGE_M[1]),
        "n_points": int(len(P)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data/realdata")
    ap.add_argument("--config-dir", default="configs/RealData")
    ap.add_argument("--scenes", nargs="*", default=None)
    args = ap.parse_args()

    scenes = args.scenes or sorted(
        d for d in os.listdir(args.root)
        if os.path.exists(os.path.join(args.root, d, "traj.txt")))
    hdr = "%-26s %10s %8s %8s %8s  %s" % (
        "scene", "法線[deg]", "y_low", "y_high", "span", "判定")
    print(hdr)
    print("-" * len(hdr))
    all_ok = True
    for s in scenes:
        cfg = os.path.join(args.config_dir, "%s.yaml" % s)
        if not os.path.exists(cfg):
            print("%-26s (config なし)" % s)
            continue
        r = check_scene(os.path.join(args.root, s), cfg)
        if "error" in r:
            print("%-26s %s" % (s, r["error"]))
            all_ok = False
            continue
        ok = r["normal_ok"] and r["span_ok"]
        all_ok &= ok
        print("%-26s %10.3f %8.2f %8.2f %8.2f  %s" % (
            s, r["floor_normal_vs_up_deg"], r["y_low"], r["y_high"], r["y_span_m"],
            "OK" if ok else "NG"))
    print()
    print("span は床から天井までの高さ。S0 実測の天井高（約 2.58 m）と合うはず。")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
