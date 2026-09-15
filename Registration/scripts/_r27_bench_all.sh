#!/bin/bash
# R27 §3 — GT-A 基準で 8 シーン × 8 手法 × 100 試行。
#
# **シーンごとに別の out_dir へ書く。** 途中で落ちても、終わったシーンは残る（R23 §4-6）。
# 手法は R27 §2-4 のとおり **baseline 4 種**（RANSAC p2p / RANSAC p2l / FGR p2p / FGR p2l）と
# **アブレーション 3 種**を含む。「11.2%」は 4 種のうちの最良値なので、4 種すべてが要る。
cd /home/student/rizu/SNI-SLAM
source /opt/miniconda/3/etc/profile.d/conda.sh
conda activate sni-slam

METHODS="proposed baseline_open3d baseline_ransac_p2l baseline_fgr baseline_fgr_p2l proposed_no_semantic proposed_no_gravity proposed_fixed_scale"

for s in room_0 room_1 room_2 office_0 office_1 office_2 office_3 office_4; do
  echo "================ $s  $(date '+%H:%M:%S') ================"
  python Registration/scripts/benchmark.py \
    --config "Registration/configs/gt_a/$s.yaml" \
    --methods $METHODS \
    --out-dir "output/Registration/gt_a/$s" 2>&1 | grep -v "RPly\|Open3D WARNING"
done
echo "================ 完了 $(date '+%H:%M:%S') ================"
