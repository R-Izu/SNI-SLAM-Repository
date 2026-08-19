#!/usr/bin/env bash
# 大きい bound のシーンについて、ckpt からメッシュを作り直し、
# precheck と 2D ラベル投影まで仕上げる。
#
# 背景: meshing.resolution が 0.01 のままだと、廊下を含む bound（例 2460 m^3）で
# 1.5 億点を超え、Mesher.get_bound_from_frames の TSDF と合わせて OOM する。
# トラッキングは完走して ckpt が残っているので、解像度と keyframe 間引きを
# 変えてメッシュだけ作り直せば 4 時間の再実行を避けられる。
set -eo pipefail
source /opt/miniconda/3/etc/profile.d/conda.sh
cd "$(dirname "$0")/../.."
conda activate sni-slam

SCENES="${*:-m3_block_b m3_block_d}"

for s in $SCENES; do
  echo "############ $s ############"
  run="output/RealData/_D/$s/run1"
  if [ ! -f "$run/mesh/final_mesh_semantic.ply" ]; then
    python -W ignore tools/realdata/mesh_from_ckpt.py \
        --config "configs/RealData/_D/$s.yaml" --run "$run" --kf-stride 4 \
        2>&1 | grep -viE "it/s\]|it\]" | tail -6
  else
    echo "mesh already exists, skip"
  fi

  python tools/realdata/precheck_scene.py \
      --mesh "$run/mesh/final_mesh_semantic.ply" \
      --traj-gt "data/realdata/$s/traj.txt" --est-poses "$run/ckpts" \
      --scene "$s" --out "$run/precheck_$s.json" 2>&1 | tail -5

  python tools/realdata/project_labels_to_mesh.py \
      --mesh "$run/mesh/final_mesh_semantic.ply" \
      --scene-dir "data/realdata/$s" --config "configs/RealData/_D/$s.yaml" \
      --out "$run/mesh/final_mesh_semantic_projected.ply" --frame-stride 8 \
      2>&1 | tail -3

  rm -rf "$run/feat_cache"
  echo
done
