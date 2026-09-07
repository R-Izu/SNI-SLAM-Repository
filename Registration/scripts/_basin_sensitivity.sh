#!/usr/bin/env bash
# (ii)/(iii) の分かれ目は「吸引域に入ったか」の閾値で決まる。
# plane_match の中央値が 0.970 m で、既定の閾値 1.0 m のすぐ内側にある。
# **閾値を1つ選んで報告すると、際どい判定を確定的に見せてしまう。** 感度を出す。
set -eo pipefail
source /opt/miniconda/3/etc/profile.d/conda.sh
cd /home/student/rizu/SNI-SLAM
conda activate sni-slam

for B in 0.5 0.8 1.0 1.5 2.0 3.0; do
  echo "########## basin = ${B} m ##########"
  python -W ignore Registration/scripts/failure_decomposition.py --basin "$B" \
      --out "/tmp/fd_${B}.json" \
      --sweeps Registration/output/diag/sweep_m3_cor_c__E2_io1.json \
               Registration/output/diag/sweep_m3_cor_a__E3_io1.json 2>&1 \
    | grep -E '^===|\(i\)|\(ii\)|\(iii\)|\(ok\)'
done
