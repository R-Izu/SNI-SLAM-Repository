"""Phase 4 sensitivity figures: label-noise and density degradation curves.

Reads the summary.json files produced by the phase-4 benchmark runs
(``--label-noise`` / ``--voxel-size`` sweeps) plus the section-C defaults, and
draws success-rate curves with Wilson 95% CI bands per scene.

Usage:
  python Registration/scripts/plot_sensitivity.py \
      --phase4-dir output/Registration/journal_phase4 \
      --sectionc-dir output/Registration/sectionC \
      --out-dir output/Registration/sectionC_analysis
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK = "#333333"
GRID = dict(color="#cccccc", linewidth=0.6, alpha=0.6)
SCENE_COLORS = {"room_0": "#2a78d6", "office_0": "#1baf7a", "office_4": "#eda100",
                "office_2": "#eda100"}

NOISE_SCENES = ("room_0", "office_0", "office_4")
NOISE_LEVELS = ((0.0, None), (0.05, "005"), (0.10, "010"), (0.20, "020"),
                (0.30, "030"), (0.50, "050"))
DENSITY_SCENES = ("room_0", "office_0", "office_2")
DENSITY_LEVELS = ((0.0, "0"), (0.02, "002"), (0.05, None), (0.10, "010"),
                  (0.20, "020"))


def _agg(path: str) -> Optional[Dict]:
    if not os.path.isfile(path):
        return None
    d = json.load(open(path))
    return d["methods"]["proposed"]["aggregate"]


def _series(scene: str, levels, phase4_dir: str, sectionc_dir: str,
            prefix: str) -> Tuple[List[float], List[float], List[float], List[float]]:
    xs, ys, lo, hi = [], [], [], []
    for level, tag in levels:
        if tag is None:   # section-C default run
            a = _agg(os.path.join(sectionc_dir, scene, "summary.json"))
        else:
            a = _agg(os.path.join(phase4_dir, f"{prefix}_{scene}_{'p' if prefix=='noise' else 'v'}{tag}",
                                  "summary.json"))
        if a is None:
            continue
        xs.append(level)
        ys.append(a["success_rate"])
        lo.append(a["success_ci_lo"])
        hi.append(a["success_ci_hi"])
    return xs, ys, lo, hi


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase4-dir", default="output/Registration/journal_phase4")
    ap.add_argument("--sectionc-dir", default="output/Registration/sectionC")
    ap.add_argument("--out-dir", default="output/Registration/sectionC_analysis")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))

    ax = axes[0]
    for scene in NOISE_SCENES:
        xs, ys, lo, hi = _series(scene, NOISE_LEVELS, args.phase4_dir,
                                 args.sectionc_dir, "noise")
        c = SCENE_COLORS[scene]
        ax.fill_between(xs, lo, hi, color=c, alpha=0.15, linewidth=0)
        ax.plot(xs, ys, color=c, linewidth=2, marker="o", markersize=5, label=scene)
    ax.set_xlabel("label noise fraction p", color=INK)
    ax.set_ylabel("success rate (proposed)", color=INK)
    ax.set_title("semantic label noise (G4-3)", color=INK)

    ax = axes[1]
    for scene in DENSITY_SCENES:
        xs, ys, lo, hi = _series(scene, DENSITY_LEVELS, args.phase4_dir,
                                 args.sectionc_dir, "density")
        c = SCENE_COLORS[scene]
        ax.fill_between(xs, lo, hi, color=c, alpha=0.15, linewidth=0)
        ax.plot(xs, ys, color=c, linewidth=2, marker="o", markersize=5, label=scene)
    ax.set_xlabel("voxel size [m] (0 = raw cloud)", color=INK)
    ax.set_title("point density / voxel sweep (G4-2)", color=INK)

    for ax in axes:
        ax.set_ylim(-0.02, 1.05)
        ax.grid(True, **GRID)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=9)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.tight_layout()
    path = os.path.join(args.out_dir, "sensitivity_curves.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
