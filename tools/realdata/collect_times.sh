#!/usr/bin/env bash
# 各シーンの SLAM 実測所要（tqdm の最終行から）を集計する。
cd "$(dirname "$0")/../.."
printf "%-12s %8s %12s %10s\n" scene frames elapsed s/frame
printf -- "----------------------------------------------\n"
tot=0
for s in m3_room_a m3_room_b m3_block_a m3_cor_b m3_block_b m3_block_d; do
  f="output/RealData/_logs/slam_${s}_run1.log"
  [ -f "$f" ] || { printf "%-12s (ログ無し)\n" "$s"; continue; }
  # ★ "Tracking Frame" を含む行だけを見る。ログ末尾には cull_mesh の進捗バーも
  #   出ており、そちらを拾うと 0.14 s/frame のような非現実的な値になる。
  line=$(tr '\r' '\n' < "$f" | grep 'Tracking Frame' \
         | grep -oE '[0-9]+/[0-9]+ \[[0-9:]+<' | tail -1)
  n=$(echo "$line" | cut -d/ -f2 | cut -d' ' -f1)
  el=$(echo "$line" | sed -E 's/.*\[([0-9:]+)<.*/\1/')
  secs=$(echo "$el" | awk -F: '{if(NF==3) print $1*3600+$2*60+$3; else print $1*60+$2}')
  tot=$((tot + secs))
  printf "%-12s %8s %12s %10.2f\n" "$s" "$n" "$el" "$(echo "$secs $n" | awk '{print $1/$2}')"
done
printf -- "----------------------------------------------\n"
printf "合計 %.1f 時間\n" "$(echo "$tot" | awk '{print $1/3600}')"
