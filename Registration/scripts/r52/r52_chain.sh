#!/bin/bash
# R52：入力作り（r52_inputs.sh）とビット一致の確認（r52_run.py bitcheck）の終わりを待ち、40/40 なら位置合わせ（320 試行、8 並列）、その後に採点。
# 40/40 でなければ止まる（指示書 §3-1）
source /opt/miniconda/3/etc/profile.d/conda.sh && conda activate sni-slam
cd ~/rizu/SNI-SLAM; L=~/rizu/r52_data/logs
until grep -q "inputs done" $L/inputs.log && [ -f Registration/output/diag/r52/bitcheck.json ]; do sleep 60; done
n=$(python -c "import json;print(sum(r[\"bit_equal\"] for r in json.load(open(\"Registration/output/diag/r52/bitcheck.json\"))[\"rows\"]))")
echo "bitcheck $n/40" >> $L/chain.log
[ "$n" = "40" ] || { echo "STOP: bitcheck not 40/40" >> $L/chain.log; exit 1; }
python Registration/scripts/r52/r52_run.py run 8 2>&1 | grep --line-buffered -v RPly > $L/run.log
python Registration/scripts/r52/r52_score.py 2>&1 | grep --line-buffered -v RPly > $L/score.log
echo "chain done" >> $L/chain.log
