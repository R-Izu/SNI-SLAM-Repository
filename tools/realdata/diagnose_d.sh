#!/usr/bin/env bash
# フォールバック D の各シーンについて、どこまで出来ているかを一覧する。
cd "$(dirname "$0")/../.."

for s in m3_room_a m3_room_b m3_block_a m3_cor_b m3_block_b m3_block_d; do
  d="output/RealData/_D/$s/run1"
  nm=$(ls "$d/mesh" 2>/dev/null | wc -l)
  nc=$(ls "$d/ckpts" 2>/dev/null | wc -l)
  last=$(ls "$d/ckpts" 2>/dev/null | tail -1)
  ate=$([ -f "$d/eval_ate.json" ] && echo yes || echo no)
  pre=$([ -f "$d/precheck_$s.json" ] && echo yes || echo no)
  printf "%-12s mesh=%-3s ckpts=%-3s last=%-12s eval_ate=%-4s precheck=%s\n" \
      "$s" "$nm" "$nc" "${last:-none}" "$ate" "$pre"
done

echo
echo "=== SLAM ログ末尾（エラー行のみ） ==="
for s in m3_cor_b m3_block_b m3_block_d; do
  f="output/RealData/_logs/slam_${s}_run1.log"
  echo "--- $s ---"
  if [ -f "$f" ]; then
    tr '\r' '\n' < "$f" | grep -iE "error|traceback|killed|out of memory|assert|Exception" | tail -5
    tr '\r' '\n' < "$f" | grep -vE "Tracking Frame|Mapping Frame|it/s\]|it\]" | tail -3
  else
    echo "(ログ無し)"
  fi
done
