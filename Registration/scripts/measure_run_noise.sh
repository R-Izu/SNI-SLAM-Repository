#!/usr/bin/env bash
# 同一コード・同一 seed で benchmark を N 回回し、**実行間のばらつき**を測る。
#
# なぜ要るか
# ----------
# `io_utils._load_slam_mesh` の `sample_points_uniformly()` は seed を取れない
# （Open3D 0.13 にこの引数が無い）ので、**実行のたびに違う 20 万点を引く**。
# したがって results.csv の数値には実行間の揺らぎが乗っている。
# **その幅を知らずに小数3桁で引用してはいけない。**
#
# 本 Vault の記録基準（CLAUDE.md「評価結果の記録基準」）が要求する
# 「run 間変動を記録する」に該当する。
set -eo pipefail
source /opt/miniconda/3/etc/profile.d/conda.sh
cd "$(dirname "$0")/../.."
conda activate sni-slam

N="${1:-5}"
TRIALS="${2:-20}"
OUT=Registration/output/run_noise
rm -rf "$OUT"; mkdir -p "$OUT"

for i in $(seq 1 "$N"); do
  echo "--- run $i / $N ---"
  python -W ignore Registration/scripts/benchmark.py \
      --config Registration/configs/replica_room0.yaml \
      --methods proposed --trials "$TRIALS" --out-dir "$OUT/run$i" 2>&1 | tail -1
done

python - "$OUT" "$N" <<'PY'
import csv, os, sys
import numpy as np
root, n = sys.argv[1], int(sys.argv[2])
rows = []
for i in range(1, n + 1):
    p = os.path.join(root, "run%d" % i, "results.csv")
    if not os.path.exists(p):
        continue
    with open(p) as f:
        for r in csv.DictReader(f):
            if r["method"] == "proposed":
                rows.append(r)
keys = ["success_rate", "direct_chamfer", "direct_inlier", "direct_rot_deg",
        "direct_trans", "med_rot_deg", "med_trans", "med_scale_ratio"]
print("\n同一コード・同一 seed を %d 回（%s 試行/回）" % (len(rows), rows[0]["robust_trials"]))
print("%-18s %10s %10s %10s %10s %9s" % ("指標", "中央値", "平均", "標準偏差", "最小", "最大"))
print("-" * 72)
rep = {}
for k in keys:
    v = np.array([float(r[k]) for r in rows if r[k] not in ("", "None")])
    if not len(v):
        continue
    print("%-18s %10.4f %10.4f %10.4f %10.4f %9.4f"
          % (k, np.median(v), v.mean(), v.std(), v.min(), v.max()))
    rep[k] = {"median": float(np.median(v)), "mean": float(v.mean()),
              "std": float(v.std()), "min": float(v.min()), "max": float(v.max()),
              "values": [float(x) for x in v]}
import json
with open(os.path.join(root, "run_noise.json"), "w") as f:
    json.dump({"n_runs": len(rows), "trials": rows[0]["robust_trials"],
               "cause": ("io_utils._load_slam_mesh の sample_points_uniformly が "
                         "seed を取れない（Open3D 0.13）ため、実行ごとに別の点を引く"),
               "metrics": rep}, f, indent=2, ensure_ascii=False)
print("\nwrote %s/run_noise.json" % root)
PY
