#!/usr/bin/env bash
# 変換済みの全シーンについて、config テンプレートを最新版で書き直す。
# bound も画像も作り直さない（conversion_report.json の値をそのまま使う）。
set -eo pipefail
source /opt/miniconda/3/etc/profile.d/conda.sh
cd "$(dirname "$0")/../.."
conda activate sni-slam

for d in data/realdata/*/; do
  s=$(basename "$d")
  [ -f "$d/conversion_report.json" ] || { echo "skip $s (not converted yet)"; continue; }
  python tools/realdata/convert_stray.py --scene "$s" --scan dummy --regen-config \
      --out data/realdata
done
echo
echo "=== sample config ==="
ls configs/RealData/
