#!/usr/bin/env bash
# フォールバック D の全シーンを **一律 0.04 m** で再メッシュし、precheck と
# 2D ラベル投影まで仕上げる。
#
# なぜ再メッシュするか:
#   小さい3シーンは meshing.resolution 0.01 で作られており、大きい3シーンは
#   その解像度では OOM して作れなかった。解像度が揃っていないと再構成の細かさが
#   シーン間で違い、比較できない（追補3 §1「凍結して全シーンに同じ config」）。
#   ckpt が残っているので SLAM の再実行は不要。
#
# cull_mesh は実行しない。下流が使うのは final_mesh_semantic.ply であって
# culled 版ではなく、全フレーム走査で時間がかかるため。
set -eo pipefail
source /opt/miniconda/3/etc/profile.d/conda.sh
cd "$(dirname "$0")/../.."
conda activate sni-slam

SCENES="${*:-m3_room_a m3_room_b m3_block_a m3_cor_b m3_block_b m3_block_d}"

for s in $SCENES; do
  echo "############ $s ############"
  run="output/RealData/_D/$s/run1"
  [ -d "$run/ckpts" ] || { echo "no ckpts, skip"; continue; }

  python -W ignore tools/realdata/mesh_from_ckpt.py \
      --config "configs/RealData/_D/$s.yaml" --run "$run" \
      --kf-stride 4 --no-cull 2>&1 | grep -viE "it/s\]|it\]|^INFO" | tail -5

  python tools/realdata/precheck_scene.py \
      --mesh "$run/mesh/final_mesh_semantic.ply" \
      --traj-gt "data/realdata/$s/traj.txt" --est-poses "$run/ckpts" \
      --scene "$s" --out "$run/precheck_$s.json" 2>&1 | tail -4

  python tools/realdata/project_labels_to_mesh.py \
      --mesh "$run/mesh/final_mesh_semantic.ply" \
      --scene-dir "data/realdata/$s" --config "configs/RealData/_D/$s.yaml" \
      --out "$run/mesh/final_mesh_semantic_projected.ply" --frame-stride 8 \
      2>&1 | tail -3

  rm -rf "$run/feat_cache"
  echo
done
