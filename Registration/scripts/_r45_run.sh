#!/bin/bash
# R45 の 1 ジョブ。引数：<mode: centroid|plan_a> <scene> [trials]
# centroid は R27 の config（gt_a/<scene>.yaml）、plan_a は gt_a_plan_a/<scene>.yaml。手法は proposed だけ。
cd /home/student/rizu/SNI-SLAM
source /opt/miniconda/3/etc/profile.d/conda.sh
conda activate sni-slam
MODE="$1"; SCENE="$2"; TRIALS="$3"
if [ "$MODE" = "centroid" ]; then CFG="Registration/configs/gt_a/${SCENE}.yaml"; else CFG="Registration/configs/gt_a_plan_a/${SCENE}.yaml"; fi
OUTD="output/Registration/r45${TRIALS:+_pre}/${MODE}/${SCENE}"
EXTRA=""; [ -n "$TRIALS" ] && EXTRA="--trials $TRIALS"
echo "commit $(git rev-parse HEAD) dirty=$(git status --porcelain | wc -l) cfg=$CFG out=$OUTD start $(date '+%F %T')"
python Registration/scripts/benchmark.py --config "$CFG" --methods proposed --out-dir "$OUTD" $EXTRA 2>&1 | grep -v "RPly\|Open3D WARNING"
echo "done $(date '+%F %T')"
