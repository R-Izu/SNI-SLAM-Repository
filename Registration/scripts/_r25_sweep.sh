#!/bin/bash
# R25 §4 の掃き出し（1回限りの補助）。既定は書き換えず、コマンドラインで上書きする。
cd ~/rizu/SNI-SLAM
source /opt/miniconda/3/etc/profile.d/conda.sh
conda activate sni-slam
for t in "$@"; do
  for si in median_axes vertical_only; do
    echo "=== $t $si ==="
    python Registration/scripts/export_stages.py --target "$t" --scale-init "$si" --no-ply 2>&1 \
      | grep -E "S2_|S5_final|候補のスコア|Error|error"
  done
done
