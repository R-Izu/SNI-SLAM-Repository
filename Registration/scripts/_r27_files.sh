#!/bin/bash
# R27 §1-3 の検査 1・3：traj.txt の有無と、同じディレクトリの配布物との比較。
cd /home/student/rizu/SNI-SLAM

echo "=== 検査1：8 シーンに traj.txt があるか ==="
for s in room_0 room_1 room_2 office_0 office_1 office_2 office_3 office_4; do
  for d in "data/replica/${s}_official" "data/replica/${s}"; do
    if [ -d "$d" ]; then
      if [ -f "$d/traj.txt" ]; then
        printf '%-28s traj.txt あり  行数 %s\n' "$d" "$(wc -l < "$d/traj.txt")"
      else
        printf '%-28s **traj.txt 無し**  中身: %s\n' "$d" "$(ls "$d" | tr '\n' ' ')"
      fi
    fi
  done
done

echo
echo "=== 検査3：更新時刻・所有者・大きさを、同じディレクトリの配布物と比べる ==="
for s in room_0 room_1 room_2 office_0 office_1 office_2 office_3 office_4; do
  d="data/replica/${s}_official"
  [ -d "$d" ] || continue
  echo "--- $d ---"
  ls -l --time-style=long-iso "$d" 2>/dev/null | head -8
  for sub in rgb depth semantic_class results; do
    if [ -d "$d/$sub" ]; then
      f=$(ls "$d/$sub" | head -1)
      ls -l --time-style=long-iso "$d/$sub/$f" 2>/dev/null | sed 's/^/    /'
      echo "    （$sub の中身 $(ls "$d/$sub" | wc -l) 件）"
    fi
  done
done

echo
echo "=== traj.txt の書式（先頭の数値の書き方）==="
for s in room_0 room_1 room_2 office_0 office_1 office_2 office_3 office_4; do
  f="data/replica/${s}_official/traj.txt"
  [ -f "$f" ] || continue
  printf '%-12s 1行目の先頭2値: %s\n' "$s" "$(head -1 "$f" | cut -d' ' -f1-2)"
  printf '%-12s 1行の値の数: %s / 最終行の末尾: %s\n' "" \
    "$(head -1 "$f" | wc -w)" "$(tail -1 "$f" | rev | cut -d' ' -f1 | rev)"
done
