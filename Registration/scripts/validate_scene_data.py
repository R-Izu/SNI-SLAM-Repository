"""Validate a rendered Replica scene directory against the SNI-SLAM loader contract.

The Replica loader (src/utils/datasets.py) requires:
  <scene>/rgb/rgb_*.png            (1200x680, 8-bit color)
  <scene>/depth/depth_*.png        (1200x680, uint16, depth = value / png_depth_scale)
  <scene>/semantic_class/semantic_class_*.png  (raw Replica class ids)
  <scene>/traj.txt                 (one flattened 4x4 c2w matrix per line, >= n_img lines)

Semantic ids are remapped through the global class list [0, 31, 37, 40, 93, 97]
(seg/semantic_classes.pkl); ids outside the list collapse to background, so this
script also reports the pixel coverage of the structural ids per scene.

Usage:
  python Registration/scripts/validate_scene_data.py data/replica/room_1_official \
      [--out validation.json] [--sample-frames 50]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from typing import Dict, List, Optional

import cv2
import numpy as np

EXPECTED_WH = (1200, 680)
STRUCTURAL_IDS = (31, 37, 40, 93, 97)  # ceiling, door, floor, wall, window
BLINDS_ID = 12  # remapped to window by the loader hack (datasets.py:103)
DEPTH_SCALE = 1000.0
SANE_DEPTH_RANGE_M = (0.05, 20.0)


def _numeric_sort(paths: List[str]) -> List[str]:
    def key(p: str):
        nums = re.findall(r"\d+", os.path.basename(p))
        return int(nums[0]) if nums else os.path.basename(p)
    return sorted(paths, key=key)


def _check_traj(path: str, n_img: int) -> Dict:
    result: Dict = {"exists": os.path.isfile(path)}
    if not result["exists"]:
        return result
    with open(path) as f:
        lines = [ln for ln in f.read().splitlines() if ln.strip()]
    result["n_lines"] = len(lines)
    result["enough_lines"] = len(lines) >= n_img
    bad = 0
    for ln in lines:
        vals = ln.split()
        if len(vals) != 16:
            bad += 1
            continue
        try:
            [float(v) for v in vals]
        except ValueError:
            bad += 1
    result["bad_lines"] = bad
    result["parses"] = bad == 0
    return result


def _sample_indices(n: int, k: int) -> np.ndarray:
    return np.unique(np.linspace(0, n - 1, min(k, n)).astype(int))


def validate_scene(scene_dir: str, sample_frames: int = 50) -> Dict:
    """Run all checks on one scene directory and return a JSON-able report."""
    report: Dict = {"scene_dir": os.path.abspath(scene_dir)}
    rgb = _numeric_sort(glob.glob(os.path.join(scene_dir, "rgb", "rgb_*.png")))
    depth = _numeric_sort(glob.glob(os.path.join(scene_dir, "depth", "depth_*.png")))
    sem = _numeric_sort(glob.glob(os.path.join(scene_dir, "semantic_class",
                                               "semantic_class_*.png")))
    report["counts"] = {"rgb": len(rgb), "depth": len(depth), "semantic_class": len(sem)}
    report["counts_match"] = len(rgb) > 0 and len(rgb) == len(depth) == len(sem)
    report["traj"] = _check_traj(os.path.join(scene_dir, "traj.txt"), len(rgb))

    if rgb:
        img = cv2.imread(rgb[0], cv2.IMREAD_UNCHANGED)
        report["rgb_resolution"] = [int(img.shape[1]), int(img.shape[0])]
        report["rgb_resolution_ok"] = (img.shape[1], img.shape[0]) == EXPECTED_WH
    if depth:
        dep = cv2.imread(depth[0], cv2.IMREAD_UNCHANGED)
        report["depth_resolution"] = [int(dep.shape[1]), int(dep.shape[0])]
        report["depth_resolution_ok"] = (dep.shape[1], dep.shape[0]) == EXPECTED_WH
        report["depth_dtype"] = str(dep.dtype)
        vals = dep[dep > 0].astype(np.float64) / DEPTH_SCALE
        if vals.size:
            report["depth_range_m"] = [float(vals.min()), float(vals.max())]
            report["depth_range_ok"] = bool(vals.min() >= SANE_DEPTH_RANGE_M[0]
                                            and vals.max() <= SANE_DEPTH_RANGE_M[1])

    if sem:
        idx = _sample_indices(len(sem), sample_frames)
        counts: Dict[int, int] = {}
        total = 0
        for i in idx:
            img = cv2.imread(sem[i], cv2.IMREAD_UNCHANGED)
            ids, cnts = np.unique(img, return_counts=True)
            for u, c in zip(ids.tolist(), cnts.tolist()):
                counts[int(u)] = counts.get(int(u), 0) + int(c)
            total += img.size
        structural = sum(counts.get(i, 0) for i in STRUCTURAL_IDS)
        known = structural + counts.get(BLINDS_ID, 0) + counts.get(0, 0)
        report["semantic"] = {
            "sampled_frames": int(len(idx)),
            "unique_ids": sorted(counts.keys()),
            "structural_pixel_share": round(structural / total, 4),
            "per_id_share": {str(i): round(counts.get(i, 0) / total, 4)
                             for i in STRUCTURAL_IDS + (BLINDS_ID,)},
            "unknown_to_background_share": round((total - known) / total, 4),
        }

    checks = [report.get("counts_match", False),
              report["traj"].get("enough_lines", False),
              report["traj"].get("parses", False),
              report.get("rgb_resolution_ok", False),
              report.get("depth_resolution_ok", False),
              report.get("depth_range_ok", False)]
    report["all_ok"] = all(checks)
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("scene_dirs", nargs="+", help="scene directories to validate")
    ap.add_argument("--out", default=None, help="write combined JSON report here")
    ap.add_argument("--sample-frames", type=int, default=50)
    args = ap.parse_args()

    reports = []
    for d in args.scene_dirs:
        rep = validate_scene(d, args.sample_frames)
        reports.append(rep)
        status = "OK " if rep["all_ok"] else "NG "
        sem = rep.get("semantic", {})
        print(f"[{status}] {d}  frames={rep['counts']}  "
              f"structural_share={sem.get('structural_pixel_share')}")
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(reports, f, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
