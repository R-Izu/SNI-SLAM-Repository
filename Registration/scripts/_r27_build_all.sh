#!/bin/bash
# R27 §2-2：8 シーンぶんの GT-A を作る。**1 シーンごとに書き出すので途中で落ちても残る。**
cd /home/student/rizu/SNI-SLAM
source /opt/miniconda/3/etc/profile.d/conda.sh
conda activate sni-slam
for s in room_0 room_1 room_2 office_0 office_1 office_2 office_3 office_4; do
  echo "================ $s ================"
  python Registration/scripts/build_gt_a.py --scene "$s" 2>&1 \
    | grep -v "RPly\|Open3D WARNING"
done
