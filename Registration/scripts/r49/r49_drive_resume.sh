#!/bin/bash
# 10-15 23:30 に Windows Update の自動の再起動で止まったので、続きから回す（済んだ試行は飛ばす）
# R49 段階1：指示書 §9 の順番で回す。**回す前に commit する。**
# 1. 並列の数の試し：合成の壁の向きからのヨーの候補なし trial 10・11（16 試行）を 4 並列（本番の置き場所）と 16 並列（probe16）で 1 回ずつ。
#    1 試行あたりの経過時間を S6 の 8 並列（92 s）と並べ、いちばん短い並列の数を以後の並列の実行に使う
# 2. 完全・クラス別なしの 1 本ずつの実行：合成（taskset 8 コア、スレッド 8）
# 3. 壁の向きからのヨーの候補なし：合成の残り
# 4. 同：実データ（並列）。完全・クラス別なし：対照・実データ（1 本ずつ）
# 5. 記述だけのもの（非構造物を入れる、実データの物理拘束なし・ラベルを使わない 4 方式）
source /opt/miniconda/3/etc/profile.d/conda.sh && conda activate sni-slam
cd ~/rizu/SNI-SLAM
L=~/rizu/r49_data/logs; mkdir -p $L
R=Registration/scripts/r49/r49_run.py
run_par() {  # 並列数 組 ログ [--out 置き場所]
  local t0=$(date +%s); python $R parallel "$@" 2>&1 | grep --line-buffered -v RPly >> $L/$3.log; local t1=$(date +%s)
  echo "$2 $1 $((t1 - t0))" >> $L/elapsed.txt
}
# 再開（10-16）：並列の数の試しは済み（4 並列 2365 s・16 並列 1535 s ／ 16 試行、S6 の 8 並列 92 s ／ 試行）→ 8 並列
# 1 試行あたりの経過（16 試行）：4 並列・16 並列と、S6 の 8 並列（7342 s / 80）
BEST=8
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
taskset -c 0-7 python $R serial synth 2>&1 | grep --line-buffered -v RPly >> $L/serial_synth.log
unset OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS
run_par $BEST nowall_synth nowall_synth
run_par $BEST nowall_real nowall_real
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
taskset -c 0-7 python $R serial bim,real 2>&1 | grep --line-buffered -v RPly >> $L/serial_bim_real.log
unset OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS
run_par $BEST descriptive descriptive
echo "all done" >> $L/elapsed.txt
