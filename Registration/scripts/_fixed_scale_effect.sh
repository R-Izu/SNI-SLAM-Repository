#!/usr/bin/env bash
# R8 §2-3-3 — 縮尺固定のバグを直したことで、`proposed_fixed_scale` の解がどれだけ動くか。
#
# フルの回し直し（R8 §9 で保留）ではなく、**入力を固定して 1 回だけ register を呼び**、
# 修正前（3e6e4b3）と修正後で返る Sim(3) を直接比べる。
# サンプリングが非決定的なので、点群を .npz に固定してから両方に読ませる。
set -eo pipefail
source /opt/miniconda/3/etc/profile.d/conda.sh
cd "$(dirname "$0")/../.."
REPO=$(pwd)
conda activate sni-slam

BASE="${1:-3e6e4b3}"
CFG="${2:-Registration/configs/replica_room0.yaml}"
OUT="$REPO/Registration/output/fixed_scale_effect"
WT=$(mktemp -d)/base
rm -rf "$OUT"; mkdir -p "$OUT"
cleanup() { git -C "$REPO" worktree remove --force "$WT" 2>/dev/null || true; }
trap cleanup EXIT

python -W ignore Registration/scripts/_regress_probe.py --config "$CFG" --dump "$OUT/input.npz"

for M in proposed proposed_fixed_scale; do
  python -W ignore Registration/scripts/_regress_probe.py --config "$CFG" \
      --load "$OUT/input.npz" --method "$M" --out "$OUT/after_$M.json" >/dev/null
done

git -C "$REPO" worktree add --detach "$WT" "$BASE" >/dev/null
ln -s "$REPO/data" "$WT/data"; ln -s "$REPO/output" "$WT/output"
cp Registration/scripts/_regress_probe.py "$WT/Registration/scripts/"
for M in proposed proposed_fixed_scale; do
  ( cd "$WT" && python -W ignore Registration/scripts/_regress_probe.py --config "$CFG" \
      --load "$OUT/input.npz" --method "$M" --out "$OUT/before_$M.json" >/dev/null )
done

python - "$OUT" "$CFG" <<'PY'
import json, os, sys
import numpy as np
sys.path.insert(0, "Registration")
from regbim.metrics import decompose_sim3
root = sys.argv[1]
print("%-22s %10s %10s %12s" % ("method", "並進差m", "回転差°", "縮尺(前→後)"))
print("-" * 58)
for m in ("proposed", "proposed_fixed_scale"):
    a = np.array(json.load(open(os.path.join(root, "after_%s.json" % m)))["T"])
    b = np.array(json.load(open(os.path.join(root, "before_%s.json" % m)))["T"])
    Ra, ta, sa = decompose_sim3(a)
    Rb, tb, sb = decompose_sim3(b)
    c = (np.trace(Ra @ Rb.T) - 1) / 2
    print("%-22s %10.4f %10.4f  %.4f→%.4f"
          % (m, float(np.linalg.norm(ta - tb)),
             float(np.degrees(np.arccos(np.clip(c, -1, 1)))), sb, sa))
print()
print("`proposed` は with_scaling=true なので上書きが起きず、差が 0 になるはず。")
print("`proposed_fixed_scale` だけが動く。それがバグの影響の大きさである。")
PY
