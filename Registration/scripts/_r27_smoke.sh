#!/bin/bash
# GT-A の疎通と所要時間の実測（5 試行）。本番の見積もりに使う。
cd /home/student/rizu/SNI-SLAM
source /opt/miniconda/3/etc/profile.d/conda.sh
conda activate sni-slam
time python Registration/scripts/benchmark.py \
  --config Registration/configs/gt_a/room_0.yaml --trials 5 \
  --methods proposed baseline_open3d baseline_fgr baseline_fgr_p2l baseline_ransac_p2l \
  --out-dir output/Registration/gt_a_smoke/room_0 2>&1 | grep -v "RPly\|Open3D WARNING"
