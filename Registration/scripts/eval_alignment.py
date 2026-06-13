"""Align a SLAM cloud to the IFC/GT reference with a chosen method and log how the
per-iteration Sim3 deviates from the frozen ``T_gt``.

This is the repeatable evaluation harness behind the ``/eval-registration`` slash
command (task A: compare "do-nothing"/baseline alignment vs the proposed
physical-constraint + semantic-ICP method, both numerically and visually).

Outputs (under ``--out``):
  - ``aligned.ply``           source cloud after the final Sim3, coloured by class
  - ``reference.ply``         the reference (GT) cloud, coloured by class (for overlay)
  - ``report.txt``            input paths + final error + per-iteration deviation table
  - ``trajectory.csv``        same table, machine-readable for plotting comparisons

Run inside the ``sni-slam`` conda env, from the repo root:
    python Registration/scripts/eval_alignment.py --method proposed --stride 1
"""

import argparse
import os
import time

import _bootstrap  # noqa: F401
import numpy as np
import open3d as o3d

from regbim import io_utils, metrics
from regbim.config import load_config, load_t_gt
from regbim.labels import LABEL_COLORS, LabeledCloud
from regbim.methods import available_methods, get_method
from regbim.trace import Tracer


def _save_colored(points: np.ndarray, labels: np.ndarray, path: str) -> None:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points))
    pcd.colors = o3d.utility.Vector3dVector(LABEL_COLORS[np.asarray(labels)] / 255.0)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    o3d.io.write_point_cloud(path, pcd)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="Registration/configs/replica_room0.yaml")
    ap.add_argument("--method", default=None,
                    help="registered method name (default: cfg.adopted_method)")
    ap.add_argument("--src", default=None,
                    help="override SLAM source mesh/cloud path")
    ap.add_argument("--dst", default=None,
                    help="override reference (IFC/GT) mesh/cloud path")
    ap.add_argument("--stride", type=int, default=1,
                    help="record the Sim3 every N ICP iterations (default 1)")
    ap.add_argument("--out", default=None,
                    help="output dir (default: Registration/output/eval/align_<method>_<ts>)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    method_name = args.method or cfg.get("adopted_method", "proposed")
    if args.src:
        cfg["source"]["mesh_path"] = args.src
    if args.dst:
        cfg["reference"]["mesh_path"] = args.dst

    out_dir = args.out or os.path.join(
        cfg["eval"]["out_dir"],
        f"align_{method_name}_{time.strftime('%y%m%d_%H%M%S')}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"methods available: {available_methods()}")
    print(f"method = {method_name}   stride = {args.stride}")

    T_gt = load_t_gt(cfg)
    if T_gt is None:
        raise SystemExit(
            "no frozen T_gt — run Registration/scripts/establish_gt.py first "
            "(deviation-from-GT needs it).")

    src = io_utils.load_source_cloud(cfg)
    dst = io_utils.load_reference_cloud(cfg)
    print("SOURCE (SLAM):", io_utils.class_counts(src))
    print("REFERENCE (IFC/GT):", io_utils.class_counts(dst))

    tracer = Tracer(stride=args.stride)
    t0 = time.time()
    T = get_method(method_name).register(src, dst, cfg, tracer=tracer)
    dt = time.time() - t0

    # Per-iteration deviation rows.
    rows = []
    for it, Ti in tracer.steps:
        e = metrics.sim3_errors(Ti, T_gt)
        rows.append((it, e["rot_deg"], e["trans"], e["scale_ratio"]))
    final = metrics.sim3_errors(T, T_gt)

    # --- aligned + reference clouds (for the visual comparison) ---------------
    aligned_pts = metrics.apply_sim3(T, src.points)
    _save_colored(aligned_pts, src.labels, os.path.join(out_dir, "aligned.ply"))
    _save_colored(dst.points, dst.labels, os.path.join(out_dir, "reference.ply"))

    # --- trajectory.csv -------------------------------------------------------
    csv_path = os.path.join(out_dir, "trajectory.csv")
    with open(csv_path, "w") as f:
        f.write("iter,rot_deg,trans_m,scale_ratio\n")
        for it, r, t, s in rows:
            f.write(f"{it},{r:.6f},{t:.6f},{s:.6f}\n")

    # --- report.txt -----------------------------------------------------------
    src_path = cfg["source"].get("mesh_path", "?")
    dst_path = cfg["reference"].get("mesh_path", "?")
    report_path = os.path.join(out_dir, "report.txt")
    with open(report_path, "w") as f:
        f.write("# Registration evaluation\n")
        f.write(f"method        : {method_name}\n")
        f.write(f"config        : {args.config}\n")
        f.write(f"source (SLAM) : {src_path}\n")
        f.write(f"reference(GT) : {dst_path}\n")
        if cfg["reference"].get("info_path"):
            f.write(f"reference info: {cfg['reference']['info_path']}\n")
        f.write(f"stride        : {args.stride}\n")
        f.write(f"runtime_sec   : {dt:.2f}\n")
        f.write(f"final error vs T_gt: rot={final['rot_deg']:.4f} deg  "
                f"trans={final['trans']:.4f} m  scale_ratio={final['scale_ratio']:.4f}\n")
        f.write("\n# per-iteration deviation from T_gt\n")
        f.write(f"{'iter':>6}  {'rot_deg':>10}  {'trans_m':>10}  {'scale_ratio':>12}\n")
        for it, r, t, s in rows:
            f.write(f"{it:>6}  {r:>10.4f}  {t:>10.4f}  {s:>12.4f}\n")

    print(f"\n=== {method_name} ===")
    print(f"time={dt:.2f}s  final: rot={final['rot_deg']:.3f}deg "
          f"trans={final['trans']:.3f}m scale_ratio={final['scale_ratio']:.3f}")
    print(f"recorded {len(rows)} steps")
    print(f"outputs -> {out_dir}/ (aligned.ply, reference.ply, report.txt, trajectory.csv)")


if __name__ == "__main__":
    main()
