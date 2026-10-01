#!/bin/bash
# R43 の対照の 1 ジョブ。引数：config 名（Registration/configs/r43_bim/<名>.yaml）
cd /home/student/rizu/SNI-SLAM
source /opt/miniconda/3/etc/profile.d/conda.sh
conda activate sni-slam
NAME="$1"
echo "commit $(git rev-parse HEAD) dirty=$(git status --porcelain | wc -l) cfg=$NAME start $(date '+%F %T')"
python Registration/scripts/benchmark.py --config "Registration/configs/r43_bim/${NAME}.yaml" --methods proposed \
  --out-dir "output/Registration/r43_bim/${NAME}" 2>&1 | grep -v "RPly\|Open3D WARNING"
echo "done $(date '+%F %T')"
