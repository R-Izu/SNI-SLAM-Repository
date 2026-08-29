"""R5 §6 — GT 位置合わせキットを作る（撮影者が CloudCompare で手動合わせするための一式）。

なぜ急ぐか
----------
GT 位置合わせは追補6 §5 の 3 であり、**4・5・6 の全部の前提**である。
そして**人手が律速**なので、こちらの都合で待たせてはいけない。

出すもの（`output/GT_alignment/`）
---------------------------------
- `source/<scene>.ply`      各シーンの TSDF メッシュ（**動かす側**）
- `reference/bim_4FL.ply`   BIM 由来の点群（**固定する側**）
- `README.md`               撮影者が単独で作業できる手順
- `T_gt/`                   変換行列の置き場（空。ここへ保存してもらう）
- `verify_gt.py`            受け取った GT を機械的に検算する（別ファイル）

設計の判断
----------
- **GT はシーンごとに1つでよい。** 参照の変種（411 / 410 / 411+410）は
  **同じ BIM 座標系の部分集合**なので、変種ごとに作り直す必要は無い。
  → **10 シーン ＝ 10 回**。30 回ではない
- source は**メッシュのまま**渡す（点群に落とさない）。CloudCompare は PLY メッシュを
  そのまま開き、`Align` の点ピックもメッシュ上で行える。面が見える方が対応点を取りやすい
- 参照は**点群**にする。IFC の三角形は室内側・室外側の両面があり、
  メッシュで渡すと室外の面が手前に見えて対応点を取りにくい

    conda activate sni-slam
    python Registration/scripts/make_gt_kit.py
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from typing import Dict, List

import numpy as np

SCENES: List[str] = ["m3_cor_c", "m3_cor_d",          # 両室＋廊下 = E1/E2 の材料。最優先
                     "m3_cor_a", "m3_cor_b",
                     "m3_block_a", "m3_block_b", "m3_block_c", "m3_block_d",
                     "m3_room_a", "m3_room_b"]
# 撮影者が特定した covers（どのスキャンがどの室を含むか）
COVERS: Dict[str, str] = {
    "m3_room_a": "411 のみ", "m3_room_b": "411 のみ",
    "m3_block_a": "411 のみ", "m3_block_b": "411 のみ",
    "m3_block_c": "411 + 410", "m3_block_d": "411 + 410",
    "m3_cor_a": "411 + 廊下", "m3_cor_b": "411 + 廊下",
    "m3_cor_c": "411 + 410 + 廊下", "m3_cor_d": "411 + 410 + 廊下",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="output/GT_alignment")
    ap.add_argument("--bim", default="Registration/output/ifc/m3_ifc_all.npz")
    args = ap.parse_args()

    repo = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "..", ".."))
    os.chdir(repo)
    import open3d as o3d

    src_dir = os.path.join(args.out, "source")
    ref_dir = os.path.join(args.out, "reference")
    for d in (src_dir, ref_dir, os.path.join(args.out, "T_gt")):
        os.makedirs(d, exist_ok=True)

    # --- source: TSDF メッシュ ------------------------------------------------
    manifest: List[Dict] = []
    for s in SCENES:
        m = "output/RealData/_TSDF/%s/run1/mesh/final_mesh_semantic.ply" % s
        if not os.path.exists(m):
            print("  %-12s メッシュ無し (%s)" % (s, m))
            manifest.append({"scene": s, "status": "missing"})
            continue
        dst = os.path.join(src_dir, "%s.ply" % s)
        shutil.copyfile(m, dst)
        mesh = o3d.io.read_triangle_mesh(dst)
        v = np.asarray(mesh.vertices)
        manifest.append({
            "scene": s, "status": "ok", "file": "source/%s.ply" % s,
            "covers": COVERS.get(s), "n_vertices": int(len(v)),
            "bbox_min_m": [round(float(x), 2) for x in v.min(axis=0)],
            "bbox_max_m": [round(float(x), 2) for x in v.max(axis=0)],
            "size_mb": round(os.path.getsize(dst) / 1e6, 1),
        })
        print("  %-12s %8s 頂点  %s" % (s, "{:,}".format(len(v)), COVERS.get(s)))

    # --- reference: BIM 点群 --------------------------------------------------
    z = np.load(args.bim, allow_pickle=False)
    pts, lab = z["points"], z["labels"]
    meta = json.loads(str(z["meta"]))
    palette = np.array([(128, 128, 128), (255, 64, 64), (255, 200, 64),
                        (180, 220, 255), (64, 200, 255), (200, 100, 255)],
                       dtype=np.float64) / 255.0
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pts)
    pc.colors = o3d.utility.Vector3dVector(palette[np.clip(lab, 0, 5)])
    if "normals" in z.files:
        pc.normals = o3d.utility.Vector3dVector(z["normals"])
    ref_path = os.path.join(ref_dir, "bim_4FL.ply")
    o3d.io.write_point_cloud(ref_path, pc)
    print("\n  reference  %s 点  %s" % ("{:,}".format(len(pts)), ref_path))

    ref_info = {
        "file": "reference/bim_4FL.ply",
        "source_ifc": meta["path"], "schema": meta["schema"],
        "storeys": meta.get("storeys_requested"),
        "spaces": "411 + 410（切り出し無し）",
        "units": "m（ifcopenshell.geom が m で返すため、IfcSIUnit の mm 係数は掛けていない）",
        "origin": "IFC のプロジェクト原点そのまま。平行移動は一切していない",
        "applied_translation": meta["applied_translation"],
        "bbox_min_m": meta["bbox_min_m"], "bbox_max_m": meta["bbox_max_m"],
        "up_axis": "+Z（床上面 z=0.510 / 天井下面 z=3.110 → 室高 2.600 m）",
        "class_counts": meta["class_counts"],
        "n_points": int(len(pts)),
    }

    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump({"scenes": manifest, "reference": ref_info}, f,
                  indent=2, ensure_ascii=False)
    print("\nwrote %s" % os.path.join(args.out, "manifest.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
