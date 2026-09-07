"""R12 §3-2 — シーンごとの $\\Delta$ を、そのシーンの幾何と並べる。

問い：**なぜ `m3_cor_c`（0.163 m）と `m3_room_b`（0.251 m）だけ大きいのか。**

並べる量（すべて既存の測定から取れる。新しい実行はしない）：
  - $\\Delta$（6基準の相互変位の中央値・最大）
  - 壁方向数と集中度（`output/RealData/_TSDF/*/run1/precheck.json`）
  - 床面積（S0 の preflight）と、参照が覆う割合の目安
  - 撮影者の目視評価
  - GT の精緻化で ICP が動かした量（手動位置合わせの不確かさ）
"""

from __future__ import annotations

import glob
import json
import os
import sys
from typing import Dict

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
os.chdir(REPO)

SCAN = {"m3_room_a": "707e94b9a5", "m3_room_b": "a499702124",
        "m3_block_a": "03d9034a38", "m3_block_b": "fdbbcc52f1",
        "m3_block_c": "b8ebcd8158", "m3_block_d": "d02315a5e8",
        "m3_cor_a": "5355665db1", "m3_cor_b": "5e2269a58d",
        "m3_cor_c": "6e4afbf6e6", "m3_cor_d": "b17452f252"}
VISUAL = {"m3_cor_a": 1, "m3_cor_b": 1, "m3_block_a": 3, "m3_block_b": 3,
          "m3_room_a": 5, "m3_room_b": 5, "m3_block_c": 7, "m3_block_d": 8}
COVERS = {"m3_room_a": "411", "m3_room_b": "411", "m3_block_a": "411",
          "m3_block_b": "411", "m3_block_c": "411+410", "m3_block_d": "411+410",
          "m3_cor_a": "411+廊下", "m3_cor_b": "411+廊下",
          "m3_cor_c": "411+410+廊下", "m3_cor_d": "411+410+廊下"}


def rj(p: str) -> Dict:
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return {}


def main() -> int:
    cs = rj("Registration/output/diag/criterion_spread.json")
    ref = rj("output/GT_alignment/refine_gt_icp.json")
    ref_by = {r["scene"]: r for r in ref} if isinstance(ref, list) else {}
    pre = {}
    for f in glob.glob("output/RealData/_preflight/preflight_*.json"):
        pre[os.path.basename(f)[10:-5]] = rj(f)

    rows = []
    for scene, blob in cs.get("scenes", {}).items():
        d = [v["p0_disp_m"] for v in blob["pairwise"].values()]
        pc = rj("output/RealData/_TSDF/%s/run1/precheck.json" % scene)
        pd = pc.get("plane_diversity") or {}
        area = (pre.get(SCAN.get(scene, ""), {}) or {}).get("alpha_shape_area_m2")
        rows.append({
            "scene": scene,
            "delta_med": float(np.median(d)), "delta_max": float(np.max(d)),
            "n_dir": pd.get("n_directions"),
            "conc": pd.get("concentration_vs_uniform"),
            "area": area, "covers": COVERS.get(scene), "visual": VISUAL.get(scene),
            "gt_icp_trans": (ref_by.get(scene) or {}).get("delta_translation_m"),
            "gt_icp_rot": (ref_by.get(scene) or {}).get("delta_rotation_deg"),
        })
    rows.sort(key=lambda r: -r["delta_med"])

    print("シーンごとの Δ（6基準の相互変位）と、そのシーンの幾何")
    print("%-12s %9s %9s %5s %7s %8s %-14s %5s %9s"
          % ("scene", "Δ中央値", "Δ最大", "壁方向", "集中度", "床m2", "covers",
             "目視", "GT-ICP m"))
    print("-" * 92)
    for r in rows:
        print("%-12s %9.3f %9.3f %5s %7s %8s %-14s %5s %9s"
              % (r["scene"], r["delta_med"], r["delta_max"],
                 r["n_dir"], "%.1f" % r["conc"] if r["conc"] else "-",
                 round(r["area"]) if r["area"] else "-", r["covers"],
                 r["visual"] if r["visual"] else "-",
                 "%.3f" % r["gt_icp_trans"] if r["gt_icp_trans"] else "-"))

    big = [r for r in rows if r["delta_med"] > 0.1]
    print("\nΔ 中央値が閾値 0.1 m を超えたシーン: %s"
          % ", ".join(r["scene"] for r in big))
    a = np.array([r["delta_med"] for r in rows])
    for key, lab in (("gt_icp_trans", "GT-ICP の並進"), ("area", "床面積"),
                     ("conc", "集中度")):
        v = [r[key] for r in rows if r.get(key) is not None]
        if len(v) == len(rows) and len(v) > 3:
            print("  Δ中央値 と %s の順位相関: %+.3f"
                  % (lab, float(np.corrcoef(
                      np.argsort(np.argsort(a)),
                      np.argsort(np.argsort(np.array(v))))[0, 1])))
    with open("Registration/output/diag/criterion_spread_table.json", "w") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print("\nwrote Registration/output/diag/criterion_spread_table.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
