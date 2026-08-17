#!/usr/bin/env bash
# 2D ラベルのメッシュ投影を、depth 許容差を変えて実行する（感度確認つき）。
# 使い方: run_projection.sh <scene> [run_no]
set -eo pipefail
source /opt/miniconda/3/etc/profile.d/conda.sh
cd "$(dirname "$0")/../.."
conda activate sni-slam

SCENE="${1:?scene required}"
RUN="${2:-1}"
MESHDIR="output/RealData/${SCENE}/run${RUN}/mesh"

for tol in 0.12 0.30; do
  echo "=== depth-tol ${tol} ==="
  python tools/realdata/project_labels_to_mesh.py \
      --mesh "${MESHDIR}/final_mesh_semantic.ply" \
      --scene-dir "data/realdata/${SCENE}" \
      --config "configs/RealData/${SCENE}.yaml" \
      --out "${MESHDIR}/final_mesh_semantic_projected_tol${tol}.ply" \
      --frame-stride 8 --depth-tol "${tol}" 2>&1 | tail -4
  echo
done
