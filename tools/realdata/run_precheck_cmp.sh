#!/usr/bin/env bash
# A（SLAM トラッキング）と D（ARKit 姿勢）の precheck を同条件で比較する。
set -eo pipefail
source /opt/miniconda/3/etc/profile.d/conda.sh
cd "$(dirname "$0")/../.."
conda activate sni-slam

echo "=== A: m3_room_a (SLAM tracking) ==="
python tools/realdata/precheck_scene.py \
    --mesh output/RealData/m3_room_a/run1/mesh/final_mesh_semantic.ply \
    --traj-gt data/realdata/m3_room_a/traj.txt \
    --est-poses output/RealData/m3_room_a/run1/ckpts \
    --scene m3_room_a_A \
    --out output/RealData/m3_room_a/run1/precheck_m3_room_a.json 2>&1 | tail -6

echo
echo "=== D: m3_room_a (ARKit poses, mapping only) ==="
python tools/realdata/precheck_scene.py \
    --mesh output/RealData/_D/m3_room_a/run1/mesh/final_mesh_semantic.ply \
    --traj-gt data/realdata/m3_room_a/traj.txt \
    --est-poses output/RealData/_D/m3_room_a/run1/ckpts \
    --scene m3_room_a_D \
    --out output/RealData/_D/m3_room_a/run1/precheck.json 2>&1 | tail -6

echo
echo "=== 参照: Replica room0_official（既知の良好） ==="
python tools/realdata/precheck_scene.py \
    --mesh output/Replica/room0_official/260310_test4/mesh/final_mesh_semantic.ply \
    --scene replica_room0_official \
    --out output/Replica/room0_official/260310_test4/precheck_ref.json 2>&1 | tail -6
