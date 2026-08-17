#!/usr/bin/env bash
# 実行中の SLAM を監視し、1 分ごとに「経過・最大RSS・現在フレーム」を1行出す。
# 異常（Traceback / OOM / Killed）を見つけたら即座に出して終了する。
# 最終メッシュができたら DONE を出して終了する。
#
# 使い方: watch_run.sh <scene> [max_minutes]
set -eo pipefail
cd "$(dirname "$0")/../.."

SCENE="${1:?scene name required}"
MAXMIN="${2:-90}"
LOG="output/RealData/_logs/slam_${SCENE}_run1.log"
MESH="output/RealData/${SCENE}/run1/mesh/final_mesh_semantic.ply"

for i in $(seq 1 "$MAXMIN"); do
  sleep 60

  if [ -f "$MESH" ]; then
    echo "DONE ${SCENE}: final_mesh_semantic.ply created after ${i} min"
    exit 0
  fi

  if [ -f "$LOG" ] && grep -qiE 'traceback|out of memory|killed|cuda error|assertionerror' "$LOG"; then
    echo "FAIL ${SCENE}:"
    grep -iE 'traceback|out of memory|killed|cuda error|assertionerror' "$LOG" | tail -3
    exit 1
  fi

  # SLAM プロセスが消えていたら（メッシュも無いのに）異常終了
  if ! pgrep -f "run.py configs/RealData/${SCENE}.yaml" > /dev/null; then
    echo "FAIL ${SCENE}: process gone at ${i} min without producing a mesh"
    tail -5 "$LOG" 2>/dev/null | tr '\r' '\n' | tail -3
    exit 1
  fi

  rss=$(ps -o rss= -C python 2>/dev/null | sort -n | tail -1)
  rss_gb=$(awk -v k="${rss:-0}" 'BEGIN{printf "%.1f", k/1048576}')
  frame=$(tr '\r' '\n' < "$LOG" 2>/dev/null | grep -oE 'Tracking Frame [0-9]+' | tail -1)
  echo "${SCENE} t=${i}min maxRSS=${rss_gb}GB ${frame:-starting}"
done

echo "TIMEOUT ${SCENE}: still running after ${MAXMIN} min"
