#!/bin/bash
# R32 §4 / R31 §4：残り 7 シーンで同じ被覆掃引。
# **見たいのは1つだけ：分離境界がシーンをまたいで安定しているか。**
cd /home/student/rizu/SNI-SLAM
source /opt/miniconda/3/etc/profile.d/conda.sh
conda activate sni-slam
for s in room_1 room_2 office_0 office_1 office_2 office_3 office_4; do
  echo "================ $s  $(date '+%H:%M:%S') ================"
  python Registration/scripts/coverage_sweep.py --scene "$s" \
    --keep 1.0 0.97 0.95 0.92 0.90 0.85 0.80 0.70 0.60 0.50 0.40 0.30 \
    --anchors="-1,-1;-1,1;1,-1;1,1" \
    --out "Registration/output/diag/coverage_sweep_${s}.json" 2>&1 \
    | grep -v "RPly\|Open3D WARNING" | tail -6
done
echo "================ 完了 $(date '+%H:%M:%S') ================"
