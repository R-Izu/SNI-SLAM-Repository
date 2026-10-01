#!/bin/bash
# R40 の 1 ジョブ。引数：config 名（Registration/configs/r40_bim/<名>.yaml）
cd /home/student/rizu/SNI-SLAM
source /opt/miniconda/3/etc/profile.d/conda.sh
conda activate sni-slam
NAME="$1"
CFG="Registration/configs/r40_bim/${NAME}.yaml"
case "$NAME" in
  E2p_centroid) METHODS="proposed proposed_no_semantic proposed_no_gravity proposed_fixed_scale baseline_open3d baseline_ransac_p2l baseline_fgr baseline_fgr_p2l" ;;
  E2p_plan_correlate) METHODS="proposed" ;;
  *) METHODS="proposed proposed_no_semantic proposed_no_gravity baseline_open3d" ;;
esac
echo "commit $(git rev-parse HEAD) dirty=$(git status --porcelain | wc -l) cfg=$CFG methods=$METHODS start $(date '+%F %T')"
python Registration/scripts/benchmark.py --config "$CFG" --methods $METHODS \
  --out-dir "output/Registration/r40_bim/${NAME}" 2>&1 | grep -v "RPly\|Open3D WARNING"
echo "done $(date '+%F %T')"
