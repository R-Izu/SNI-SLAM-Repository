#!/bin/bash
cd /home/student/rizu/SNI-SLAM
source /opt/miniconda/3/etc/profile.d/conda.sh
conda activate sni-slam
python Registration/scripts/coverage_sweep.py --scene room_0 \
  --keep 1.0 0.97 0.95 0.92 0.90 0.85 0.80 0.70 0.60 0.50 0.40 0.30 \
  --anchors="-1,-1;-1,1;1,-1;1,1" 2>&1 | grep -v "RPly\|Open3D WARNING"
