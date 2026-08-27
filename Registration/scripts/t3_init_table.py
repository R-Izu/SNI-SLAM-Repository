"""T3 の下見（初期スケール）を、被覆率 × モードの表にする。

初期スケールは摂動に依存しないので、benchmark の 100 試行を待たずに読める。
**この段階で「vertical_only が被覆に不変か」は判定できる**（成功率はまだ言えない）。
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

root = sys.argv[1] if len(sys.argv) > 1 else "Registration/output/t3_coverage"
d = json.load(open(os.path.join(root, "init_scale.json")))

scenes, levels = [], ["100", "075", "050", "030"]
for k in d:
    s = k.split("__")[0]
    if s not in scenes:
        scenes.append(s)

print("初期スケール誤差（seed されたスケール / 真のスケール。1.0 が正解）")
print("%-10s %-14s %8s %8s %8s %8s   %s"
      % ("scene", "mode", "100%", "75%", "50%", "30%", "100%→30% の変化"))
print("-" * 82)
summary = {}
for s in scenes:
    for mode in ("median_axes", "vertical_only"):
        vals = []
        for lv in levels:
            hit = [v for k, v in d.items()
                   if k.startswith(s + "__cov" + lv) and k.endswith(mode)]
            vals.append(hit[0].get("init_scale_error_ratio") if hit else None)
        if any(v is None for v in vals):
            continue
        drift = vals[-1] - vals[0]
        print("%-10s %-14s %8.3f %8.3f %8.3f %8.3f   %+.3f"
              % (s, mode, *vals, drift))
        summary.setdefault(mode, []).append({"scene": s, "vals": vals, "drift": drift})
    print()

print("=" * 82)
for mode, rows in summary.items():
    dr = np.array([r["drift"] for r in rows])
    err100 = np.abs(np.array([r["vals"][0] for r in rows]) - 1.0)
    err30 = np.abs(np.array([r["vals"][3] for r in rows]) - 1.0)
    print("%-14s 100%%→30%% の変化: 平均 %+.3f / 最大絶対 %.3f"
          % (mode, dr.mean(), np.abs(dr).max()))
    print("%-14s   |誤差| 100%%: 中央値 %.3f / 30%%: 中央値 %.3f"
          % ("", np.median(err100), np.median(err30)))

n = len(summary.get("vertical_only", []))
flat = sum(1 for r in summary.get("vertical_only", []) if abs(r["drift"]) < 0.01)
print("\nvertical_only が被覆率に対して実質不変（|変化| < 0.01）だったシーン: %d/%d"
      % (flat, n))
better30 = sum(1 for a, b in zip(summary.get("median_axes", []),
                                 summary.get("vertical_only", []))
               if abs(b["vals"][3] - 1.0) < abs(a["vals"][3] - 1.0))
better100 = sum(1 for a, b in zip(summary.get("median_axes", []),
                                  summary.get("vertical_only", []))
                if abs(b["vals"][0] - 1.0) < abs(a["vals"][0] - 1.0))
print("30%% 被覆で vertical_only の方が真値に近いシーン: %d/%d" % (better30, n))
print("100%% 被覆で vertical_only の方が真値に近いシーン: %d/%d" % (better100, n))
