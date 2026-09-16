#!/bin/bash
# 非有限ガードを入れたうえで room_1 を回し直す（唯一落ちたシーン）。
cd /home/student/rizu/SNI-SLAM
source /opt/miniconda/3/etc/profile.d/conda.sh
conda activate sni-slam
echo "================ room_1 再実行 $(date '+%H:%M:%S') ================"
python Registration/scripts/benchmark.py \
  --config Registration/configs/gt_a/room_1.yaml \
  --methods proposed baseline_open3d baseline_ransac_p2l baseline_fgr baseline_fgr_p2l \
            proposed_no_semantic proposed_no_gravity proposed_fixed_scale \
  --out-dir output/Registration/gt_a/room_1 2>&1 | grep -v "RPly\|Open3D WARNING"
echo "================ 完了 $(date '+%H:%M:%S') ================"
