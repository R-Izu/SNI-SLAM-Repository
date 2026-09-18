#!/bin/bash
# R31 §3：探索半径だけを変えて、同じ被覆掃引を回す。
# 既定 0.30 は R30 §3-2 で測定済み。追加は 0.15 と 0.60。
cd /home/student/rizu/SNI-SLAM
source /opt/miniconda/3/etc/profile.d/conda.sh
conda activate sni-slam
rm -f Registration/output/diag/_smoke.json
KEEP="1.0 0.97 0.95 0.92 0.90 0.85 0.80 0.70 0.60 0.50 0.40 0.30"
ANCH="-1,-1;-1,1;1,-1;1,1"
for r in 0.15 0.60; do
  echo "================ max_corr_dist = $r  $(date '+%H:%M:%S') ================"
  python Registration/scripts/coverage_sweep.py --scene room_0 \
    --keep $KEEP --anchors="$ANCH" --max-corr-dist $r \
    --out "Registration/output/diag/coverage_sweep_r${r}.json" 2>&1 \
    | grep -v "RPly\|Open3D WARNING" | tail -20
done
echo "================ 完了 $(date '+%H:%M:%S') ================"
