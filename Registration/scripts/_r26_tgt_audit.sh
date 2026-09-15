#!/bin/bash
# R26 §1 の付随確認：sectionC の 8 シーンに凍結 T_gt があったのかを、当時のログから見る。
cd /home/student/rizu/SNI-SLAM
for f in output/Registration/sectionC/*_benchmark.log; do
  n=$(basename "$f" _benchmark.log)
  if grep -q "no frozen T_gt" "$f"; then
    echo "$n : 凍結 T_gt なし → 成功率は自己一貫性のみ"
  else
    echo "$n : 凍結 T_gt あり"
  fi
done
echo "--- 実際のファイルの有無 ---"
for n in room_0 room_1 room_2 office_0 office_1 office_2 office_3 office_4; do
  p="output/Registration/sectionC/$n/T_gt.json"
  [ "$n" = "room_0" ] && p="Registration/output/eval/T_gt.json"
  if [ -f "$p" ]; then echo "$n : $p あり"; else echo "$n : $p 無し"; fi
done
