"""Extract labelled point clouds from SLAM output and the reference, and save
colour-coded PLYs for visual sanity-checking (acceptance criterion #1)."""

import argparse
import os

import _bootstrap  # noqa: F401
import numpy as np
import open3d as o3d

from regbim import io_utils
from regbim.config import load_config
from regbim.labels import LABEL_COLORS, LabeledCloud


def save_colored(cloud: LabeledCloud, path: str) -> None:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud.points)
    pcd.colors = o3d.utility.Vector3dVector(LABEL_COLORS[cloud.labels] / 255.0)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    o3d.io.write_point_cloud(path, pcd)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out_dir", default="Registration/output/labels")
    args = ap.parse_args()
    cfg = load_config(args.config)

    src = io_utils.load_source_cloud(cfg)
    ref = io_utils.load_reference_cloud(cfg)
    print("SOURCE (SLAM):", io_utils.class_counts(src))
    print("REFERENCE (BIM-surrogate):", io_utils.class_counts(ref))

    save_colored(src, os.path.join(args.out_dir, "source_labeled.ply"))
    save_colored(ref, os.path.join(args.out_dir, "reference_labeled.ply"))
    # Structural-only clouds (what the rotation step relies on).
    structural = [1, 3, 5]  # wall, floor, ceiling
    save_colored(src.subset(np.isin(src.labels, structural)),
                 os.path.join(args.out_dir, "source_structural.ply"))
    save_colored(ref.subset(np.isin(ref.labels, structural)),
                 os.path.join(args.out_dir, "reference_structural.ply"))
    print(f"saved colour-coded clouds to {args.out_dir}/")


if __name__ == "__main__":
    main()
