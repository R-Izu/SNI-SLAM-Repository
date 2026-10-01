#!/bin/bash
# R39 §5 段階1 — BIM 由来の source で 100 試行。引数：E2|E3 centroid|plan_correlate
# centroid：proposed（既定）・no_semantic・no_gravity・fixed_scale・baseline 4 種
# plan_correlate：proposed（案A）のみ
cd /home/student/rizu/SNI-SLAM
source /opt/miniconda/3/etc/profile.d/conda.sh
conda activate sni-slam
REF="$1"; MODE="$2"
CFG="Registration/configs/r39_bim/bim_${REF}_${MODE}.yaml"
if [ "$MODE" = "centroid" ]; then
  METHODS="proposed proposed_no_semantic proposed_no_gravity proposed_fixed_scale baseline_open3d baseline_ransac_p2l baseline_fgr baseline_fgr_p2l"
else
  METHODS="proposed"
fi
echo "commit $(git rev-parse HEAD) dirty=$(git status --porcelain | wc -l) cfg=$CFG methods=$METHODS start $(date '+%F %T')"
python Registration/scripts/benchmark.py --config "$CFG" --methods $METHODS \
  --out-dir "output/Registration/r39_bim/${REF}_${MODE}" 2>&1 | grep -v "RPly\|Open3D WARNING"
echo "done $(date '+%F %T')"
