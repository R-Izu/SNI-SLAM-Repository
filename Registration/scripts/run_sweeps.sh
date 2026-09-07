#!/usr/bin/env bash
# R11 §2-1 / R12 §4-1 — 2シーン × 参照2水準 × モード2種 × seed 20 本。
# 失敗の3分類（R11 §3）に必要な候補ごとの回転と ICP 後の解も記録する。
set -eo pipefail
source /opt/miniconda/3/etc/profile.d/conda.sh
cd /home/student/rizu/SNI-SLAM
conda activate sni-slam

for S in m3_cor_c__E2 m3_cor_a__E3; do
  for IO in 1 0; do
    echo "### $S inner_only=$IO ###"
    python -u -W ignore Registration/scripts/seed_sweep.py \
        --n-seeds 20 --inner-only "$IO" \
        --config "Registration/configs/realdata/$S.yaml" \
        --out "Registration/output/diag/sweep_${S}_io${IO}.json" \
        > "output/sweep_${S}_io${IO}.log" 2>&1
    tail -12 "output/sweep_${S}_io${IO}.log"
  done
done
echo "=== all done ==="
