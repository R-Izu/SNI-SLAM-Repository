import trimesh
import numpy as np
import json

# ── 旧52色のカラーマップ（変更前の Mesher.py decode_segmap と同じ配列）──
label_colors = np.array([
      (0,0,0),(174,199,232),(152,223,138),(31,119,180),(255,187,120),(188,189,34),
      (140,86,75),(255,152,150),(214,39,40),(197,176,213),(148,103,189),
      (196,156,148),(23,190,207),(178,76,76),(247,182,210),(66,188,102),
      (219,219,141),(140,57,197),(202,185,52),(51,176,203),(200,54,131),
      (92,193,61),(78,71,183),(172,114,82),(255,127,14),(91,163,138),
      (153,98,156),(140,153,101),(158,218,229),(100,125,154),(178,127,135),
      (120,185,128),(146,111,194),(44,160,44),(112,128,144),(96,207,209),
      (227,119,194),(213,92,176),(94,106,211),(82,84,163),(100,85,144),
      (100,218,200),(128,0,0),(0,128,0),(128,128,0),(0,0,128),
      (128,0,128),(0,128,128),(128,128,128),(64,0,0),(192,0,0),
      (192,128,0),(64,0,128)
  ], dtype=np.uint8)

  # ── 旧 semantic_classes（GitHub から確認した変更前の配列）──
semantic_classes = np.array([
      0, 3, 7, 8,10,11,12,13,14,15,16,17,18,19,20,22,23,26,29,31,
     34,35,37,40,44,47,52,54,56,59,60,61,62,63,64,65,70,71,76,78,
     79,80,82,83,87,88,91,92,93,95,97,98
  ])

  # ── Replica クラス名マップ（info_semantic.json から）──
with open("data/replica/room_0/habitat/info_semantic.json") as f:
      info = json.load(f)
id_to_name = {c["id"]: c["name"] for c in info["classes"]}
id_to_name[0] = "void"

# ── 旧メッシュ読み込み（WSLからWindowsのDドライブは /mnt/d/ 経由）──
mesh = trimesh.load(
    "/mnt/d/rizu/SNI-SLAM/output/Replica/room0_official/test/mesh/00440_mesh_sem.ply"
)
vertex_colors = mesh.visual.vertex_colors[:, :3]  # RGBのみ

  # ── 全ユニーク色を集計し対応表を表示 ──
unique, counts = np.unique(vertex_colors, axis=0, return_counts=True)
print(f"{'RGB':>20}  {'頂点数':>8}  {'index':>6}  {'Replica ID':>10}  クラス名")
print("-" * 65)
for rgb, cnt in sorted(zip(unique, counts), key=lambda x: -x[1]):
      diffs = np.sum(np.abs(label_colors.astype(int) - rgb.astype(int)), axis=1)
      idx = int(np.argmin(diffs))
      replica_id = int(semantic_classes[idx]) if idx < len(semantic_classes) else -1
      name = id_to_name.get(replica_id, "?")
      r, g, b_ = int(rgb[0]), int(rgb[1]), int(rgb[2])
      print(f"RGB({r:3},{g:3},{b_:3})  {cnt:>8}  {idx:>6}  {replica_id:>10}  {name}")