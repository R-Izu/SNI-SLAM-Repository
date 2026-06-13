"""Run a single registration method on SLAM -> reference and report metrics."""

import argparse
import time

import _bootstrap  # noqa: F401

from regbim import io_utils, metrics
from regbim.config import load_config, load_t_gt
from regbim.methods import available_methods, get_method


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--method", default=None, help=f"one of the registered methods")
    args = ap.parse_args()
    cfg = load_config(args.config)
    method_name = args.method or cfg.get("adopted_method", "proposed")
    print(f"methods available: {available_methods()}")

    src = io_utils.load_source_cloud(cfg)
    dst = io_utils.load_reference_cloud(cfg)

    t0 = time.time()
    T = get_method(method_name).register(src, dst, cfg)
    dt = time.time() - t0

    R, t, s = metrics.decompose_sim3(T)
    inl = metrics.class_inlier_ratio(src, dst, T, cfg["semantic_icp"]["max_corr_dist"])
    ch = metrics.chamfer_distance(metrics.apply_sim3(T, src.points), dst.points)
    print(f"\n=== {method_name} ===")
    print(f"time={dt:.2f}s  scale={s:.4f}  inlier_ratio={inl:.3f}  chamfer={ch:.4f}")

    T_gt = load_t_gt(cfg)
    if T_gt is not None:
        err = metrics.sim3_errors(T, T_gt)
        print(f"vs T_gt:  rot={err['rot_deg']:.2f} deg  trans={err['trans']:.3f} m  "
              f"scale_ratio={err['scale_ratio']:.3f}")
    else:
        print("(no frozen T_gt yet; run establish_gt.py to enable error-vs-GT)")


if __name__ == "__main__":
    main()
