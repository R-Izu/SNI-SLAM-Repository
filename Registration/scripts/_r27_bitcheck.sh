#!/bin/bash
# 非有限ガードを足す前後で、**有限な経路がビット一致すること**を示す（S4 の作法）。
#
# room_0 は旧コードで 100 試行を回し終えている。新コードで先頭 10 試行を回し直し、
# trials.csv の該当行を突き合わせる。摂動列は手法ごとに seed で決まるので、
# 試行 i は必ず同じ摂動になる。**1 行でも違えば、ガードが既存の数値を動かしている。**
cd /home/student/rizu/SNI-SLAM
source /opt/miniconda/3/etc/profile.d/conda.sh
conda activate sni-slam
python Registration/scripts/benchmark.py \
  --config Registration/configs/gt_a/room_0.yaml --trials 10 \
  --methods proposed baseline_open3d baseline_ransac_p2l baseline_fgr baseline_fgr_p2l \
            proposed_no_semantic proposed_no_gravity proposed_fixed_scale \
  --out-dir output/Registration/gt_a_bitcheck/room_0 2>&1 | grep -v "RPly\|Open3D WARNING" | tail -14
