#!/usr/bin/env bash
# 最大級のシーン（8000 frame 超・bound 40 m 級）を、keyframe を強めに間引いて再メッシュする。
#
# Mesher.get_bound_from_frames は voxel_length 7.8 mm の TSDF に keyframe を全て積む。
# m3_cor_d は 2113 keyframe あり、stride 4（528 枚）でもメモリを使い切った。
# keyframe は連続して大きく重なるので、stride を上げても被覆はほとんど落ちない。
#
# set -e は使わない（1 シーンが落ちても次を試したい）。
source /opt/miniconda/3/etc/profile.d/conda.sh
cd "$(dirname "$0")/../.."
conda activate sni-slam

STRIDE="${STRIDE:-10}"
SCENES="${*:-m3_cor_d m3_cor_c}"

for s in $SCENES; do
  echo "############ $s (kf-stride $STRIDE) ############"
  run="output/RealData/_D/$s/run1"

  if [ ! -f "$run/mesh/final_mesh_semantic.ply" ]; then
    python -W ignore tools/realdata/mesh_from_ckpt.py \
        --config "configs/RealData/_D/$s.yaml" --run "$run" \
        --kf-stride "$STRIDE" --no-cull 2>&1 | grep -viE "it/s\]|it\]|^INFO" | tail -4
  fi

  if [ -f "$run/mesh/final_mesh_semantic.ply" ]; then
    python tools/realdata/precheck_scene.py \
        --mesh "$run/mesh/final_mesh_semantic.ply" \
        --traj-gt "data/realdata/$s/traj.txt" --est-poses "$run/ckpts" \
        --scene "$s" --out "$run/precheck_$s.json" 2>&1 | tail -4
    python tools/realdata/project_labels_to_mesh.py \
        --mesh "$run/mesh/final_mesh_semantic.ply" \
        --scene-dir "data/realdata/$s" --config "configs/RealData/_D/$s.yaml" \
        --out "$run/mesh/final_mesh_semantic_projected.ply" --frame-stride 8 2>&1 | tail -2
  else
    echo "!! $s: メッシュ生成に失敗（さらに stride を上げるか解像度を落とす）"
  fi
  rm -rf "$run/feat_cache"
  echo
done
