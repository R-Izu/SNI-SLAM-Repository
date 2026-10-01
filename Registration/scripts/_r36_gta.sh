#!/bin/bash
# R36 §4-3 — GT-A 8 シーンで `proposed`（既定の centroid 経路）に rotation_release を足して 100 試行ずつ。
# 比較先は R27 §3 の 800/800。config は元の GT-A config に rotation_release だけを足したもの。
# シーンごとに別の out_dir へ書く（途中で落ちても終わったシーンは残る）。
cd /home/student/rizu/SNI-SLAM
source /opt/miniconda/3/etc/profile.d/conda.sh
conda activate sni-slam
mkdir -p Registration/configs/gt_a_release
for s in room_0 room_1 room_2 office_0 office_1 office_2 office_3 office_4; do
  python - "$s" <<'PY'
import sys, yaml
s = sys.argv[1]
c = yaml.safe_load(open("Registration/configs/gt_a/%s.yaml" % s))
c.setdefault("proposed", {})["rotation_release"] = {"enabled": True, "max_iter": 10}
yaml.safe_dump(c, open("Registration/configs/gt_a_release/%s.yaml" % s, "w"),
               allow_unicode=True, sort_keys=False)
PY
done
for s in room_0 room_1 room_2 office_0 office_1 office_2 office_3 office_4; do
  echo "================ $s  $(date '+%H:%M:%S') ================"
  python Registration/scripts/benchmark.py \
    --config "Registration/configs/gt_a_release/$s.yaml" \
    --methods proposed \
    --out-dir "output/Registration/gt_a_release/$s" 2>&1 | grep -v "RPly\|Open3D WARNING"
done
echo "================ 完了 $(date '+%H:%M:%S') ================"
