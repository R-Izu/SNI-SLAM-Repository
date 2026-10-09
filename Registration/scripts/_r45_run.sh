#!/bin/bash
# R45 の 1 ジョブ。引数：<mode: centroid|plan_a> <scene> [trials]
# 本番（R45 §6 (b)）：centroid は gt_a_centroid_seeded/<scene>.yaml、plan_a は gt_a_plan_a/<scene>.yaml（どちらも reference.seed: 0）。
# trials を渡すと試走（出力は r45_pre/）。手法は proposed だけ。
cd /home/student/rizu/SNI-SLAM
source /opt/miniconda/3/etc/profile.d/conda.sh
conda activate sni-slam
MODE="$1"; SCENE="$2"; TRIALS="$3"
if [ "$MODE" = "centroid" ]; then CFG="Registration/configs/gt_a_centroid_seeded/${SCENE}.yaml"; else CFG="Registration/configs/gt_a_plan_a/${SCENE}.yaml"; fi
OUTD="output/Registration/r45${TRIALS:+_pre}/${MODE}/${SCENE}"
EXTRA=""; [ -n "$TRIALS" ] && EXTRA="--trials $TRIALS"
echo "commit $(git rev-parse HEAD) dirty=$(git status --porcelain | wc -l) cfg=$CFG out=$OUTD start $(date '+%F %T')"
python Registration/scripts/benchmark.py --config "$CFG" --methods proposed --out-dir "$OUTD" $EXTRA 2>&1 | grep -v "RPly\|Open3D WARNING"
echo "done $(date '+%F %T')"
