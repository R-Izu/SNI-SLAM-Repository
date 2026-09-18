#!/bin/bash
# 既定 0.30 を、対応数の計測を足したコードで回し直す（3 半径を同じ列で比べるため）。
cd /home/student/rizu/SNI-SLAM
source /opt/miniconda/3/etc/profile.d/conda.sh
conda activate sni-slam
python Registration/scripts/coverage_sweep.py --scene room_0 \
  --keep 1.0 0.97 0.95 0.92 0.90 0.85 0.80 0.70 0.60 0.50 0.40 0.30 \
  --anchors="-1,-1;-1,1;1,-1;1,1" --max-corr-dist 0.30 \
  --out Registration/output/diag/coverage_sweep_r0.30.json 2>&1 \
  | grep -v "RPly\|Open3D WARNING" | tail -6
