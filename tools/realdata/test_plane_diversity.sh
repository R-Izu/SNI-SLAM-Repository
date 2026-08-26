#!/usr/bin/env bash
# plane_diversity の閾値を相対値に変えた効果を、全シーン＋Replica 参照で確認する。
# 追補4 §1 の一般則: 指標を変えたら既知の良好な結果に当てて値域を確かめる。
set -eo pipefail
source /opt/miniconda/3/etc/profile.d/conda.sh
cd "$(dirname "$0")/../.."
conda activate sni-slam

run_one () {  # <label> <mesh> <out>
  printf "%-34s " "$1"
  python tools/realdata/precheck_scene.py --mesh "$2" --scene "$1" --out "$3" 2>&1 \
      | grep "plane diversity" | sed 's/plane diversity *//'
}

echo "=== SNI-SLAM 意味場のメッシュ ==="
for s in m3_room_a m3_room_b m3_block_a m3_block_b m3_block_d m3_cor_a m3_cor_b m3_block_c; do
  run_one "$s" "output/RealData/_D/$s/run1/mesh/final_mesh_semantic.ply" \
          "output/RealData/_D/$s/run1/precheck_$s.json"
done

echo
echo "=== 2Dラベル投影後のメッシュ（こちらが本番の経路） ==="
for s in m3_room_a m3_room_b m3_block_a m3_block_b m3_block_d m3_cor_a m3_cor_b m3_block_c; do
  m="output/RealData/_D/$s/run1/mesh/final_mesh_semantic_projected.ply"
  [ -f "$m" ] || { echo "$s (投影メッシュ無し)"; continue; }
  run_one "${s}_proj" "$m" "output/RealData/_D/$s/run1/precheck_projected.json"
done

echo
echo "=== 参照: Replica room0_official（既知の良好） ==="
run_one "replica_room0" \
        "output/Replica/room0_official/260310_test4/mesh/final_mesh_semantic.ply" \
        "output/Replica/room0_official/260310_test4/precheck_ref.json"
