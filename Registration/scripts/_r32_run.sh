#!/bin/bash
cd /home/student/rizu/SNI-SLAM
source /opt/miniconda/3/etc/profile.d/conda.sh
conda activate sni-slam
python Registration/scripts/plan_a_compare.py 2>&1 | grep -v "RPly\|Open3D WARNING"
