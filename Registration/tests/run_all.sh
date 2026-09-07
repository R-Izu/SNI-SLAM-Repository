#!/usr/bin/env bash
# 全テストを走らせ、合否を1行ずつ出す。
# tests/ 配下は `synthetic` を相対 import するため、リポジトリ直下の
# Registration/ を PYTHONPATH に置く必要がある（ここで一度だけ設定する）。
cd "$(dirname "$0")/.." || exit 1
# shellcheck disable=SC1091
source /opt/miniconda/3/etc/profile.d/conda.sh && conda activate sni-slam
export PYTHONPATH=.
rc=0
for f in tests/test_*.py; do
  if out=$(python -W ignore "$f" 2>&1); then
    n=$(printf '%s\n' "$out" | grep -c '  OK   ')
    printf 'PASS    %-40s (%s checks)\n' "$f" "$n"
  else
    printf '**FAIL**  %-40s\n' "$f"
    printf '%s\n' "$out" | tail -5
    rc=1
  fi
done
exit "$rc"
