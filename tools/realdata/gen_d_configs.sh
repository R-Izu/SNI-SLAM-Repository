#!/usr/bin/env bash
# 変換済み全シーンについて、フォールバック D 用の config（func.use_gt_pose: True）を
# configs/RealData/_D/ に生成する。bound も画像も作り直さない。
set -eo pipefail
source /opt/miniconda/3/etc/profile.d/conda.sh
cd "$(dirname "$0")/../.."
conda activate sni-slam

mkdir -p configs/RealData/_D
for d in data/realdata/*/; do
  s=$(basename "$d")
  case "$s" in *_t[0-9]*) continue;; esac      # 探索用の部分列シーンは飛ばす
  [ -f "$d/conversion_report.json" ] || { echo "skip $s (not converted)"; continue; }
  python tools/realdata/convert_stray.py --scene "$s" --scan dummy \
      --regen-config --use-gt-pose --out data/realdata --config-dir configs/RealData/_D
done
echo
ls configs/RealData/_D/
