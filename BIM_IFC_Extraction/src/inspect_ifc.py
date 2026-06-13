# inspect_ifc.py
import argparse
import ifcopenshell
import sys
import os
from pathlib import Path
import numpy as np
import open3d as o3d
import ifcopenshell.geom

DENSITY = 500  # 1m²あたりの点数

LABEL_MAP = {
    "IfcWall":             (0, [255,  80,  80]),
    "IfcWallStandardCase": (0, [255,  80,  80]),  # IFC2x3互換
    "IfcDoor":             (1, [ 80, 255,  80]),
    "IfcSlab":             (2, [ 80,  80, 255]),
    "IfcWindow":           (3, [255, 255,  80]),
    "IfcCovering":         (4, [255,  80, 255]),
    "IfcColumn":           (5, [200, 100,  50]),
}

def ifc_to_mesh(shape):
    verts = np.array(shape.geometry.verts).reshape(-1, 3)
    faces = np.array(shape.geometry.faces).reshape(-1, 3)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices  = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    return mesh

def sample_mesh(mesh, label_id, rgb):
    area = mesh.get_surface_area()
    n    = max(int(area * DENSITY), 10)
    # ✅ open3d 0.13.0 対応: init_factor を明示
    pcd  = mesh.sample_points_poisson_disk(
        number_of_points=n, init_factor=5
    )
    pts    = np.asarray(pcd.points, dtype=np.float32)
    labels = np.full(len(pts), label_id, dtype=np.int32)
    colors = np.tile(np.array(rgb, dtype=np.uint8), (len(pts), 1))
    return pts, labels, colors

def parse_args():
    parser = argparse.ArgumentParser(
        description="IFCファイルをラベル付き点群（NPZ/PLY）に変換する"
    )
    parser.add_argument("ifc_path", help="入力IFCファイルのパス")
    parser.add_argument(
        "-o", "--output-dir",
        default=None,
        help="出力ディレクトリ（省略時は入力ファイルと同じディレクトリ）",
    )
    parser.add_argument(
        "--density", type=int, default=DENSITY,
        help=f"サンプリング密度 点/m² (デフォルト: {DENSITY})",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    ifc_path = Path(args.ifc_path)
    if not ifc_path.exists():
        print(f"エラー: ファイルが見つかりません: {ifc_path}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.output_dir) if args.output_dir else ifc_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    stem       = ifc_path.stem
    output_npz = out_dir / f"{stem}_labeled_pointcloud.npz"
    output_ply = out_dir / f"{stem}_labeled_pointcloud.ply"

    ifc      = ifcopenshell.open(str(ifc_path))
    settings = ifcopenshell.geom.settings()
    settings.set(settings.USE_WORLD_COORDS, True)
    settings.set(settings.WELD_VERTICES,    True)

    all_pts, all_labels, all_colors = [], [], []

    for ifc_class, (label_id, rgb) in LABEL_MAP.items():
        elems = ifc.by_type(ifc_class)
        if not elems:
            continue
        print(f"[{ifc_class:25s}] {len(elems):4d} 要素")

        for elem in elems:
            try:
                shape = ifcopenshell.geom.create_shape(settings, elem)
                mesh  = ifc_to_mesh(shape)
                if len(mesh.triangles) == 0:
                    continue
                pts, labels, colors = sample_mesh(mesh, label_id, rgb)
                all_pts.append(pts)
                all_labels.append(labels)
                all_colors.append(colors)
            except Exception:
                continue

    if not all_pts:
        print("エラー: 点群データが生成されませんでした", file=sys.stderr)
        sys.exit(1)

    xyz    = np.vstack(all_pts)
    labels = np.concatenate(all_labels)
    colors = np.vstack(all_colors)

    print(f"\n合計: {len(xyz):,} 点")
    for lid, name in [(0,"wall"),(1,"door"),(2,"slab"),
                      (3,"window"),(4,"ceiling"),(5,"column")]:
        print(f"  label {lid} ({name}): {(labels==lid).sum():,}")

    np.savez_compressed(str(output_npz), xyz=xyz, label=labels, rgb=colors)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(
        colors.astype(np.float64) / 255.0
    )
    o3d.io.write_point_cloud(str(output_ply), pcd)
    print(f"保存完了: {output_npz}, {output_ply}")

if __name__ == "__main__":
    main()
