#!/usr/bin/env bash
# 追補4 §5 — 融合方式の対照実験（SNI-SLAM のマッピング vs Open3D の TSDF 統合）。
#
# 同じ姿勢（ARKit）・同じ深度（conf>=1 かつ <=5m）で融合し、同じ指標で比較する。
# 追補4 §1 の一般則に従い、**Replica 参照にも同じ比較を当てて値域を確かめる**。
set -eo pipefail
source /opt/miniconda/3/etc/profile.d/conda.sh
cd "$(dirname "$0")/../.."
conda activate sni-slam

SCENE="${1:-m3_room_a}"
TSDF_RUN="output/RealData/_TSDF/$SCENE/run1"

echo "############ TSDF 融合: $SCENE ############"
python tools/realdata/tsdf_fusion.py \
    --scene-dir "data/realdata/$SCENE" \
    --config "configs/RealData/_D/$SCENE.yaml" \
    --out "$TSDF_RUN/mesh/final_mesh_semantic.ply" 2>&1 | tail -4

echo
echo "--- precheck (TSDF) ---"
python tools/realdata/precheck_scene.py \
    --mesh "$TSDF_RUN/mesh/final_mesh_semantic.ply" \
    --scene "${SCENE}_TSDF" --out "$TSDF_RUN/precheck.json" 2>&1 | tail -4

echo
echo "--- 2Dラベル投影 (TSDF) ---"
python tools/realdata/project_labels_to_mesh.py \
    --mesh "$TSDF_RUN/mesh/final_mesh_semantic.ply" \
    --scene-dir "data/realdata/$SCENE" --config "configs/RealData/_D/$SCENE.yaml" \
    --out "$TSDF_RUN/mesh/final_mesh_semantic_projected.ply" --frame-stride 8 2>&1 | tail -3

echo
echo "############ 参照: Replica room0_official に TSDF を当てる ############"
REF_RUN="output/RealData/_TSDF/_replica_room0/run1"
python tools/realdata/tsdf_fusion.py \
    --scene-dir "data/replica/room_0_official" \
    --config "configs/Replica/room0_official.yaml" \
    --out "$REF_RUN/mesh/final_mesh_semantic.ply" 2>&1 | tail -4

echo
echo "--- precheck (Replica + TSDF) ---"
python tools/realdata/precheck_scene.py \
    --mesh "$REF_RUN/mesh/final_mesh_semantic.ply" \
    --scene "replica_room0_TSDF" --out "$REF_RUN/precheck.json" 2>&1 | tail -4
