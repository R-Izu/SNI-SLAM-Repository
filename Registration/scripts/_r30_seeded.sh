#!/bin/bash
# R30 §4 手順3：シードを入れた状態で 2 回回し、(a) 決定的になったか (b) 値が
# シード無し 10 実行の分布のどこに来るか を見る。
cd /home/student/rizu/SNI-SLAM
source /opt/miniconda/3/etc/profile.d/conda.sh
conda activate sni-slam
python - <<'PY'
import yaml, os
os.chdir("/home/student/rizu/SNI-SLAM")
for s in ("room_0",):
    p = "Registration/configs/gt_a/%s.yaml" % s
    c = yaml.safe_load(open(p))
    c["reference"] = dict(c["reference"], seed=0)
    hdr = "".join(l for l in open(p) if l.startswith("#"))
    with open("Registration/configs/gt_a/%s_seeded.yaml" % s, "w") as f:
        f.write(hdr)
        f.write("# R30 §4 手順3：参照の標本をシード固定した版（比較用。既定は変えない）\n")
        yaml.safe_dump(c, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
print("wrote seeded config")
PY
for i in 1 2; do
  echo "================ シード固定 $i / 2 $(date '+%H:%M:%S') ================"
  python Registration/scripts/benchmark.py \
    --config Registration/configs/gt_a/room_0_seeded.yaml \
    --methods proposed \
    --out-dir "output/Registration/gt_a_refseeded/run_$i" 2>&1 \
    | grep -v "RPly\|Open3D WARNING" | tail -3
done
