#!/bin/bash
# R37 §5 — GT-A を同じ commit で off / on。経路は centroid（R36 と同じ）、8 シーン × 100 試行。
# off = 元の GT-A config（configs/gt_a）、on = rotation_release を足した config（configs/gt_a_release、R36 と同じ）。
# 摂動は config の eval.seed で決まる（R27 と同一）。シーンごとに別の out_dir へ書く。
cd /home/student/rizu/SNI-SLAM
source /opt/miniconda/3/etc/profile.d/conda.sh
conda activate sni-slam
echo "commit $(git rev-parse HEAD)  dirty=$(git status --porcelain | wc -l)"
for s in room_0 room_1 room_2 office_0 office_1 office_2 office_3 office_4; do
  for mode in off on; do
    if [ "$mode" = off ]; then c="Registration/configs/gt_a/$s.yaml"; else c="Registration/configs/gt_a_release/$s.yaml"; fi
    echo "================ $s $mode  $(date '+%H:%M:%S') ================"
    python Registration/scripts/benchmark.py --config "$c" --methods proposed \
      --out-dir "output/Registration/gt_a_r37_$mode/$s" 2>&1 | grep -v "RPly\|Open3D WARNING"
  done
done
echo "================ 完了 $(date '+%H:%M:%S') ================"
