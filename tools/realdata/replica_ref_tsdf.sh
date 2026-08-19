#!/usr/bin/env bash
# 追補4 §1 の一般則: 新しい指標・新しい比較は、既知の良好な Replica 参照に当てて
# 値域を確かめる。融合方式の比較（§5）の参照値をここで取る。
set -eo pipefail
source /opt/miniconda/3/etc/profile.d/conda.sh
cd "$(dirname "$0")/../.."
conda activate sni-slam

R=output/RealData/_TSDF/_replica_room0/run1

echo "--- Replica + TSDF: 2Dラベル投影 ---"
python tools/realdata/project_labels_to_mesh.py \
    --mesh "$R/mesh/final_mesh_semantic.ply" \
    --scene-dir data/replica/room_0_official \
    --config configs/Replica/room0_official.yaml \
    --out "$R/mesh/final_mesh_semantic_projected.ply" --frame-stride 8 2>&1 | tail -3

echo
echo "--- Replica + TSDF: precheck（投影後）---"
python tools/realdata/precheck_scene.py \
    --mesh "$R/mesh/final_mesh_semantic_projected.ply" \
    --scene replica_room0_TSDF_projected \
    --out "$R/precheck_projected.json" 2>&1 | grep -E "class fracs|gravity|plane"

echo
echo "--- 参考: Replica + SNI-SLAM（不可侵成果物）に投影を当てた場合 ---"
S=output/Replica/room0_official/260310_test4
python tools/realdata/project_labels_to_mesh.py \
    --mesh "$S/mesh/final_mesh_semantic.ply" \
    --scene-dir data/replica/room_0_official \
    --config configs/Replica/room0_official.yaml \
    --out "output/RealData/_TSDF/_replica_room0/snislam_projected.ply" --frame-stride 8 \
    2>&1 | tail -3
