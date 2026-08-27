#!/usr/bin/env bash
# 追補6 §4-2 (b) — SNI-SLAM のメッシャで OOM した m3_cor_c / m3_cor_d を
# Open3D の TSDF 統合で作る。デバッグより先にこちらを試す、という main の指示。
#
# なぜ通ると期待できるか
# ----------------------
# SNI-SLAM の Mesher は bound 全体に **密なグリッド**を張る（m3_cor_c は 44.99 x 3.46 x 27.17 m
# を 0.04 m 刻み ＝ 6600 万点）。対して ScalableTSDFVolume は **観測された表面の周りにだけ**
# ボクセルブロックを確保するので、必要メモリは bound の体積ではなく実際の表面積で決まる。
# 廊下のように「広いが中身が空」なシーンでは、この差が効くはず。
#
# 追補6 §4 補足に従い CPU と GPU を分けて記録する（Open3D は C++ 側で確保するので
# Python の tracemalloc には映らない → /usr/bin/time -v の Maximum resident set size を使う）。
set -eo pipefail
source /opt/miniconda/3/etc/profile.d/conda.sh
cd "$(dirname "$0")/../.."
conda activate sni-slam

SCENES="${*:-m3_cor_d m3_cor_c}"

for s in $SCENES; do
  run="output/RealData/_TSDF/$s/run1"
  mesh="$run/mesh/final_mesh_semantic.ply"
  echo "############################ $s ############################"
  mkdir -p "$run/mesh"

  # voxel は SNI-SLAM 側の meshing.resolution と同じ 0.04 m（追補3 §1 の設定凍結）
  /usr/bin/time -v python -W ignore tools/realdata/tsdf_fusion.py \
      --scene-dir "data/realdata/$s" \
      --config "configs/RealData/_D/$s.yaml" \
      --out "$mesh" --voxel 0.04 --frame-stride 2 \
      2> >(grep -E "Maximum resident|Elapsed \(wall" >&2) || { echo "$s: TSDF 失敗"; continue; }

  echo "--- 2Dラベル投影 ---"
  python -W ignore tools/realdata/project_labels_to_mesh.py \
      --mesh "$mesh" --scene-dir "data/realdata/$s" \
      --config "configs/RealData/_D/$s.yaml" \
      --out "$run/mesh/final_mesh_semantic_projected.ply" --frame-stride 8 2>&1 | tail -6

  # ★ precheck は**投影後**に当てる。TSDF が出す生メッシュの頂点色は写真の RGB であって
  #   6クラスのパレットではない。生メッシュに当てると色を最近傍でパレットに丸めるので、
  #   ラベルもそこから出る重力軸も無意味になる（実際 background 81% / wall 0.13% になり、
  #   重力軸が Y ではなく Z に出て「傾き 89.8°」という嘘の値が出た）。
  echo "--- precheck（投影後のメッシュに対して）---"
  python -W ignore tools/realdata/precheck_scene.py \
      --mesh "$run/mesh/final_mesh_semantic_projected.ply" \
      --scene "${s}_TSDF" --out "$run/precheck.json" 2>&1 | tail -5
  echo
done
echo "=== done ==="
