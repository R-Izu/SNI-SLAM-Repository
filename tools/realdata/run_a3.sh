#!/usr/bin/env bash
# A-3 をやり直す。**カリング後**のメッシュで、Replica 参照と実データを同条件で比較する。
#
# 最初は final_mesh_semantic.ply（カリング前）で測ってしまい、
# 「実データのマップが破綻している」と誤って結論しかけた。
# カリング前のメッシュは bound 全体の等値面で未観測領域を大量に含むため、
# accuracy が大きく出るのは当然であり、**既知の良好な Replica 参照でも同じ値になる**。
# cull_mesh が作る final_mesh_color_culled.ply が比較すべき対象。
set -eo pipefail
source /opt/miniconda/3/etc/profile.d/conda.sh
cd "$(dirname "$0")/../.."
conda activate sni-slam

run_eval () {  # <label> <run> <scene-dir> <config> <poses>
  echo "=== $1 ==="
  python tools/realdata/eval_map_vs_depth.py \
      --run "$2" --scene-dir "$3" --config "$4" --poses "$5" \
      --mesh "$2/mesh/final_mesh_color_culled.ply" \
      --out "$2/map_vs_depth_culled_$5.json" 2>&1 | tail -6
  echo
}

run_eval "Replica room0_official（既知の良好な参照）" \
    output/Replica/room0_official/260310_test4 \
    data/replica/room_0_official configs/Replica/room0_official.yaml gt

run_eval "実データ A: m3_room_a（SLAM トラッキング）" \
    output/RealData/m3_room_a/run1 \
    data/realdata/m3_room_a configs/RealData/m3_room_a.yaml est

run_eval "実データ D: m3_room_a（ARKit 姿勢・マッピングのみ）" \
    output/RealData/_D/m3_room_a/run1 \
    data/realdata/m3_room_a configs/RealData/_D/m3_room_a.yaml est
