"""追補6 §6 受入条件 — SNI-SLAM のマッピング vs Open3D TSDF を同じ指標で並べる。

追補4 §1 の一般則に従い、**Replica 参照（既知の良好な結果）にも同じ比較を当てて値域を出す**。
文献の一般論を対照の代わりにしない。

    conda activate sni-slam
    python tools/realdata/fusion_table.py
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import numpy as np

SCENES = ["m3_room_a", "m3_room_b", "m3_block_a", "m3_block_b", "m3_block_c",
          "m3_block_d", "m3_cor_a", "m3_cor_b", "m3_cor_c", "m3_cor_d"]
SCAN = {"m3_room_a": "707e94b9a5", "m3_room_b": "a499702124",
        "m3_block_a": "03d9034a38", "m3_block_b": "fdbbcc52f1",
        "m3_block_c": "b8ebcd8158", "m3_block_d": "d02315a5e8",
        "m3_cor_a": "5355665db1", "m3_cor_b": "5e2269a58d",
        "m3_cor_c": "6e4afbf6e6", "m3_cor_d": "b17452f252"}
# 撮影者の目視評価（良い順）。数値の対照が無い指標では人間の目視がその代わりになる。
VISUAL = {"m3_cor_a": 1, "m3_cor_b": 1, "m3_block_a": 3, "m3_block_b": 3,
          "m3_room_a": 5, "m3_room_b": 5, "m3_block_c": 7, "m3_block_d": 8}


def rj(p: str) -> Dict:
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return {}


def row_for(scene: str, root: str, area: Optional[float]) -> Optional[Dict]:
    run = os.path.join(root, scene, "run1")
    proj = rj(os.path.join(run, "mesh", "final_mesh_semantic_projected_report.json"))
    if not proj or not proj.get("n_vertices"):
        return None
    pre = rj(os.path.join(run, "precheck.json"))
    tsdf = rj(os.path.join(run, "mesh", "final_mesh_semantic_tsdf_report.json"))
    nv = int(proj["n_vertices"])
    cls = proj.get("class_frac_voted") or proj.get("class_frac") or {}
    return {
        "n_vertices": nv,
        "verts_per_m2": round(nv / area) if area else None,
        "depth_support": proj.get("vertices_with_votes_frac"),
        "gravity_tilt_deg": pre.get("gravity_tilt_deg"),
        "plane_diversity": pre.get("plane_diversity"),
        "elapsed_min": round(tsdf["elapsed_s"] / 60, 1) if tsdf.get("elapsed_s") else None,
        "class_frac": {k: round(v, 3) for k, v in cls.items()},
    }


def main() -> int:
    pre_area: Dict[str, float] = {}
    import glob
    for f in glob.glob("output/RealData/_preflight/preflight_*.json"):
        d = rj(f)
        pre_area[os.path.basename(f)[10:-5]] = d.get("alpha_shape_area_m2")

    out: List[Dict] = []
    hdr = "%-12s %-9s %11s %9s %10s %8s %7s" % (
        "scene", "融合", "頂点数", "頂点/m2", "深度支持率", "重力傾き", "目視")
    print(hdr)
    print("-" * len(hdr))
    for s in SCENES:
        a = pre_area.get(SCAN[s])
        for tag, root in (("SNI-SLAM", "output/RealData/_D"),
                          ("TSDF", "output/RealData/_TSDF")):
            r = row_for(s, root, a)
            if r is None:
                print("%-12s %-9s %11s" % (s, tag, "メッシュ無し"))
                out.append({"scene": s, "fusion": tag, "status": "no_mesh",
                            "floor_area_m2": a})
                continue
            print("%-12s %-9s %11s %9s %10.3f %8s %7s"
                  % (s, tag, "{:,}".format(r["n_vertices"]),
                     "{:,}".format(r["verts_per_m2"]) if r["verts_per_m2"] else "-",
                     r["depth_support"],
                     "-" if r["gravity_tilt_deg"] is None
                     else "%.2f" % r["gravity_tilt_deg"],
                     VISUAL.get(s, "-")))
            out.append({"scene": s, "fusion": tag, "status": "ok",
                        "floor_area_m2": a, "visual_rank": VISUAL.get(s), **r})

    # --- Replica 参照（既知の良好な結果）での値域 ---
    print("\n[参照] Replica room0_official — 既知の良好な結果での値域")
    for tag, root in (("SNI-SLAM", "output/Replica/room0_official/260310_test4"),
                      ("TSDF", "output/RealData/_TSDF/_replica_room0/run1")):
        proj = rj(os.path.join(root, "mesh",
                               "final_mesh_semantic_projected_report.json"))
        if not proj:
            print("  %-9s 投影レポート無し (%s)" % (tag, root))
            continue
        print("  %-9s 頂点 %s / 深度支持率 %.3f"
              % (tag, "{:,}".format(proj["n_vertices"]),
                 proj["vertices_with_votes_frac"]))
        out.append({"scene": "_replica_room0", "fusion": tag, "status": "ok",
                    "n_vertices": proj["n_vertices"],
                    "depth_support": proj["vertices_with_votes_frac"]})

    # --- 対応の取れるシーンだけで差を要約 ---
    pair = {}
    for r in out:
        if r["status"] != "ok" or r["scene"].startswith("_"):
            continue
        pair.setdefault(r["scene"], {})[r["fusion"]] = r
    both = {k: v for k, v in pair.items() if len(v) == 2}
    if both:
        ds = [(v["TSDF"]["depth_support"], v["SNI-SLAM"]["depth_support"])
              for v in both.values()]
        nv = [(v["TSDF"]["n_vertices"], v["SNI-SLAM"]["n_vertices"])
              for v in both.values()]
        print("\n両方そろう %d シーンでの差" % len(both))
        print("  深度支持率  TSDF 中央値 %.3f / SNI-SLAM 中央値 %.3f  "
              "（TSDF が高いシーン %d/%d）"
              % (np.median([a for a, _ in ds]), np.median([b for _, b in ds]),
                 sum(a > b for a, b in ds), len(ds)))
        print("  頂点数      TSDF 中央値 %s / SNI-SLAM 中央値 %s  （比 %.2f 倍）"
              % ("{:,}".format(int(np.median([a for a, _ in nv]))),
                 "{:,}".format(int(np.median([b for _, b in nv]))),
                 np.median([a for a, _ in nv]) / np.median([b for _, b in nv])))

    os.makedirs("output/RealData", exist_ok=True)
    with open("output/RealData/fusion_table.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("\nwrote output/RealData/fusion_table.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
