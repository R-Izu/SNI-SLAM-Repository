"""Establish and freeze the reference transform T_gt (SLAM -> reference).

Because the SLAM reconstruction frame and the raw Replica frame are related
non-trivially (NICE-SLAM re-rendered sequence + axis flip), there is no analytic
GT. We register once with the chosen method, save an overlay for visual
verification, and freeze the result as the evaluation reference (the same way the
research notes treat LiDAR as a 'reference', not absolute truth).
"""

import argparse
import os

import _bootstrap  # noqa: F401
import numpy as np
import open3d as o3d

from regbim import io_utils, metrics
from regbim.config import load_config, save_t_gt
from regbim.labels import LABEL_COLORS
from regbim.methods import get_method


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--method", default=None, help="method to establish GT (default: adopted_method)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    method_name = args.method or cfg.get("adopted_method", "proposed")

    src = io_utils.load_source_cloud(cfg)
    dst = io_utils.load_reference_cloud(cfg)
    print(f"establishing T_gt with method '{method_name}' ...")
    T = get_method(method_name).register(src, dst, cfg)

    R, t, s = metrics.decompose_sim3(T)
    inl = metrics.class_inlier_ratio(src, dst, T, cfg["semantic_icp"]["max_corr_dist"])
    ch = metrics.chamfer_distance(metrics.apply_sim3(T, src.points), dst.points)
    print(f"  scale={s:.4f}  inlier_ratio={inl:.3f}  chamfer={ch:.4f}")

    path = save_t_gt(cfg, T)
    print(f"  T_gt saved to {path}")

    # Overlay for visual verification: registered source (by label colour) + reference (grey).
    out_dir = cfg["eval"]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    moved = io_utils.to_o3d(metrics.transform_cloud(src, T))
    moved.colors = o3d.utility.Vector3dVector(LABEL_COLORS[src.labels] / 255.0)
    ref_pcd = io_utils.to_o3d(dst)
    ref_pcd.paint_uniform_color([0.6, 0.6, 0.6])
    o3d.io.write_point_cloud(os.path.join(out_dir, "gt_overlay_source.ply"), moved)
    o3d.io.write_point_cloud(os.path.join(out_dir, "gt_overlay_reference.ply"), ref_pcd)
    print(f"  overlay written to {out_dir}/gt_overlay_*.ply (inspect alignment)")


if __name__ == "__main__":
    main()
