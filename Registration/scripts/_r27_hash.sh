#!/bin/bash
# R27 §1-3 検査5の代用：**手元に残っている配布物そのもの**と照合する。
# ネットワーク不要。**ハッシュが一致すれば、自作ではなく配布物の写しである。**
cd /home/student/rizu/SNI-SLAM

echo "=== room_1：ダウンロードした zip の中身と照合 ==="
unzip -p data/downloads/room_1.zip room_1/traj.txt | sha256sum | sed 's/-/  (zip の中)/'
sha256sum data/replica/room_1_official/traj.txt

echo
echo "=== room_2 / office_*：vmap 配布物（traj_w_c.txt）と照合 ==="
for s in room_2 office_0 office_1 office_2 office_3 office_4; do
  a="data/downloads/vmap_extract/vmap/$s/imap/00/traj_w_c.txt"
  b="data/replica/${s}_official/traj.txt"
  if [ -f "$a" ] && [ -f "$b" ]; then
    ha=$(sha256sum "$a" | cut -c1-16)
    hb=$(sha256sum "$b" | cut -c1-16)
    if [ "$ha" = "$hb" ]; then
      printf '%-10s **一致**  %s\n' "$s" "$ha"
    else
      printf '%-10s **不一致** 配布 %s / 配置 %s\n' "$s" "$ha" "$hb"
    fi
  else
    printf '%-10s 照合できない（配布物 %s）\n' "$s" "$([ -f "$a" ] && echo あり || echo 無し)"
  fi
done

echo
echo "=== room_0：照合先の配布物が手元にあるか ==="
ls -l --time-style=long-iso data/downloads/*.zip 2>/dev/null
find data/downloads -maxdepth 6 \( -name 'traj*.txt' -o -name 'room_0*' \) 2>/dev/null | head
echo "room_0 の traj.txt:"
ls -l --time-style=long-iso data/replica/room_0_official/traj.txt
sha256sum data/replica/room_0_official/traj.txt
