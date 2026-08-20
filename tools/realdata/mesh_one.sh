#!/usr/bin/env bash
# 1 シーンだけ ckpt から再メッシュし、その間の最大 RSS を記録する。
# 使い方: mesh_one.sh <scene> [kf_stride] [resolution]
source /opt/miniconda/3/etc/profile.d/conda.sh
cd "$(dirname "$0")/../.."
conda activate sni-slam

s="${1:?scene required}"
stride="${2:-10}"
res="${3:-}"
run="output/RealData/_D/$s/run1"

resarg=""
[ -n "$res" ] && resarg="--resolution $res"

( while true; do
    r=$(ps -eo rss --sort=-rss | sed -n 2p)
    echo "  [max rss $(echo "$r" | awk '{printf "%.1f", $1/1048576}') GB]"
    sleep 60
  done ) &
watcher=$!
trap 'kill $watcher 2>/dev/null' EXIT

python -W ignore tools/realdata/mesh_from_ckpt.py \
    --config "configs/RealData/_D/$s.yaml" --run "$run" \
    --kf-stride "$stride" --no-cull $resarg 2>&1 | grep -viE 'it/s\]|it\]'
