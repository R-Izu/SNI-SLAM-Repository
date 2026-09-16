#!/bin/bash
# R30 §4 手順1：**シードを入れる前に、参照の標本ばらつきを測る。**
#
# いま `_load_replica_reference` はシード無しなので、**同じコマンドを繰り返すだけで
# 参照の標本だけが変わる**。source は points_ply で決定的、摂動列は seed 固定なので、
# **run 間で動くのは参照の標本だけ**である。
#
# 対象は proposed × room_0（R30 §4）。10 実行。
cd /home/student/rizu/SNI-SLAM
source /opt/miniconda/3/etc/profile.d/conda.sh
conda activate sni-slam
for i in $(seq 1 10); do
  echo "================ 参照標本 $i / 10  $(date '+%H:%M:%S') ================"
  python Registration/scripts/benchmark.py \
    --config Registration/configs/gt_a/room_0.yaml \
    --methods proposed \
    --out-dir "output/Registration/gt_a_refvar/run_$i" 2>&1 \
    | grep -v "RPly\|Open3D WARNING" | tail -4
done
echo "================ 完了 $(date '+%H:%M:%S') ================"
