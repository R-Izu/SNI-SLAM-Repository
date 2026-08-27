#!/usr/bin/env bash
# 融合はやり直さず、precheck だけ**投影後のメッシュ**に当て直す。
# 併せて Replica 参照の SNI-SLAM 側にも投影を当て、対照の対を揃える（追補4 §1）。
set -eo pipefail
source /opt/miniconda/3/etc/profile.d/conda.sh
cd "$(dirname "$0")/../.."
conda activate sni-slam

for s in m3_room_a m3_room_b m3_block_a m3_block_b m3_block_c m3_block_d \
         m3_cor_a m3_cor_b m3_cor_c m3_cor_d; do
  run="output/RealData/_TSDF/$s/run1"
  m="$run/mesh/final_mesh_semantic_projected.ply"
  [ -f "$m" ] || { echo "$s: 投影メッシュ無し"; continue; }
  printf "%-12s " "$s"
  python -W ignore tools/realdata/precheck_scene.py \
      --mesh "$m" --scene "${s}_TSDF" --out "$run/precheck.json" 2>&1 \
    | grep -E "重力|gravity" | head -1
done

echo
echo "=== Replica 参照: SNI-SLAM 側にも投影を当てて対を揃える ==="
REF=output/Replica/room0_official/260310_test4
if [ -f "$REF/mesh/final_mesh_semantic.ply" ]; then
  python -W ignore tools/realdata/project_labels_to_mesh.py \
      --mesh "$REF/mesh/final_mesh_semantic.ply" \
      --scene-dir data/replica/room_0_official \
      --config configs/Replica/room0_official.yaml \
      --out "$REF/mesh/final_mesh_semantic_projected.ply" --frame-stride 8 2>&1 | tail -4
else
  echo "  $REF/mesh/final_mesh_semantic.ply が無い"
fi
