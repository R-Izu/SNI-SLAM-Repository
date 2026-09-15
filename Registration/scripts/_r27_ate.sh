#!/bin/bash
# R27 §1-4：同じ config の run ごとの ATE を並べる。**同じ量を比べているかを確かめる。**
cd /home/student/rizu/SNI-SLAM
echo "run                                        rmse    median      max   単位   姿勢数   mesh"
for f in output/Replica/room0_official/*/eval_ate.json; do
  d=$(dirname "$f")
  python - "$f" "$d" <<'PY'
import json, os, sys
f, d = sys.argv[1], sys.argv[2]
try:
    j = json.load(open(f))
except Exception as e:
    print("%-40s 読めない %s" % (d, e)); raise SystemExit
m = os.path.join(d, "mesh", "final_mesh_semantic.ply")
print("%-40s %8.2f %8.2f %8.2f  %4s %7s  %s"
      % (os.path.basename(d),
         j.get("absolute_translational_error.rmse", float("nan")),
         j.get("absolute_translational_error.median", float("nan")),
         j.get("absolute_translational_error.max", float("nan")),
         j.get("units", "?"), j.get("compared_pose_pairs"),
         "あり" if os.path.exists(m) else "無し"))
PY
done
