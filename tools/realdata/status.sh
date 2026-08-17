#!/usr/bin/env bash
# 実行中パイプラインの進捗を一目で見るための小道具。
# Windows 側から `wsl -d gdep-U22 -- bash <path>/status.sh` で叩く
# （外側シェルに $ を食われないよう、変数展開は全てこのファイルの中で完結させる）。
cd "$(dirname "$0")/../.."

echo "=== processes ==="
pgrep -fa "convert_stray|gen_labels|run\.py|run_s1|run_batch" | grep -v pgrep || echo "(none)"

echo
echo "=== converted scenes ==="
for d in data/realdata/*/; do
  [ -d "$d" ] || continue
  s=$(basename "$d")
  nr=$(ls "$d/rgb" 2>/dev/null | wc -l)
  nd=$(ls "$d/depth" 2>/dev/null | wc -l)
  ns=$(ls "$d/semantic_class" 2>/dev/null | wc -l)
  sz=$(du -sh "$d" 2>/dev/null | cut -f1)
  printf "  %-28s rgb=%-6s depth=%-6s semantic=%-6s %s\n" "$s" "$nr" "$nd" "$ns" "$sz"
done

echo
echo "=== slam outputs ==="
for d in output/RealData/*/run*/; do
  [ -d "$d" ] || continue
  nc=$(ls "$d/ckpts" 2>/dev/null | wc -l)
  nm=$(ls "$d/mesh" 2>/dev/null | wc -l)
  printf "  %-44s ckpts=%-5s mesh=%-4s\n" "$d" "$nc" "$nm"
done

echo
echo "=== gpu ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader

echo
echo "=== disk ==="
df -h /mnt/d / | tail -2
