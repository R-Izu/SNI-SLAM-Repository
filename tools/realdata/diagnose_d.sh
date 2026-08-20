#!/usr/bin/env bash
# フォールバック D の各シーンについて、どこまで出来ているかを一覧する。
cd "$(dirname "$0")/../.."

SCENES="${*:-m3_room_a m3_room_b m3_block_a m3_cor_b m3_block_b m3_block_d m3_cor_a m3_block_c m3_cor_d m3_cor_c}"

for s in $SCENES; do
  d="output/RealData/_D/$s/run1"
  nm=$(ls "$d/mesh" 2>/dev/null | wc -l)
  nc=$(ls "$d/ckpts" 2>/dev/null | wc -l)
  last=$(ls "$d/ckpts" 2>/dev/null | tail -1)
  sem=$([ -f "$d/mesh/final_mesh_semantic.ply" ] && echo yes || echo NO)
  prj=$([ -f "$d/mesh/final_mesh_semantic_projected.ply" ] && echo yes || echo NO)
  pre=$([ -f "$d/precheck_$s.json" ] && echo yes || echo NO)
  printf "%-12s ckpts=%-3s last=%-12s semantic=%-4s projected=%-4s precheck=%s\n" \
      "$s" "$nc" "${last:-none}" "$sem" "$prj" "$pre"
done
