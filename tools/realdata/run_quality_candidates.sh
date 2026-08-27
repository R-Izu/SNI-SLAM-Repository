#!/usr/bin/env bash
# 品質指標の候補を全シーンで同一条件で測る。
#
# 深度支持率（分母＝メッシュ頂点数）は床面積と rho=-0.929 で交絡した。
# メッシュ体積が大きいほど偽の等値面も増えるため、分母が膨らむのが原因。
# そこで **分母を観測側に置いた指標**を測る:
#
#   completion : 観測点 -> 最近傍のメッシュ頂点 の距離
#                「見えているものが再構成されているか」。分母は観測点数なので
#                シーンの広さで自動的に増えず、サイズ交絡を受けにくい
#
# 全シーンで frame-stride / pix-stride / voxel を揃える（観測密度を揃えるため。
# eval_map_vs_depth.py を撤回した理由がまさに密度の不一致だった）。
set -eo pipefail
source /opt/miniconda/3/etc/profile.d/conda.sh
cd "$(dirname "$0")/../.."
conda activate sni-slam

for s in m3_room_a m3_room_b m3_block_a m3_block_b m3_block_c m3_block_d m3_cor_a m3_cor_b; do
  run="output/RealData/_D/$s/run1"
  [ -f "$run/mesh/final_mesh_semantic.ply" ] || { echo "$s: メッシュ無し"; continue; }
  printf "%-12s " "$s"
  python tools/realdata/eval_map_vs_depth.py \
      --run "$run" --scene-dir "data/realdata/$s" \
      --config "configs/RealData/_D/$s.yaml" --poses est \
      --frame-stride 20 --pix-stride 4 --voxel 0.02 \
      --out "$run/map_vs_depth_fixed.json" 2>&1 \
    | grep -E "^completion" | sed 's/completion *//'
done
