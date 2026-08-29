#!/usr/bin/env bash
# R3 T2 の受入条件 —「`scale_init: median_axes`（既定）で既存結果が完全に再現する」
#
# ★ benchmark.py での突き合わせは**この目的には使えない**ことが分かった。
#   `io_utils._load_slam_mesh` が `mesh.sample_points_uniformly()` を seed 無しで呼んでおり
#   （Open3D 0.13 にはこの API の seed 引数が無い）、**実行ごとに違う 20 万点を引く**。
#   実測：同一コード（HEAD）同士でも direct_rot_deg が 0.227 と 0.556 に分かれた
#   （成功率は 0.9 で一致、chamfer も 0.5311 / 0.5301 でほぼ一致）。
#   つまり**このパイプラインは元々ビット再現しない**。私の変更とは無関係の性質である。
#
# そこで「同じ入力に同じ出力が出るか」を**入力を固定して**測る：
#   1. 点群を一度だけサンプリングして .npz に保存する
#   2. **変更前のコミット**と**現在の作業ツリー**の両方で、その .npz を読んで register() を呼ぶ
#   3. 返ってきた Sim(3) を**厳密に**比較する
# register() 自体は決定的（rotation.py:58 が default_rng(0)）なので、これは
# 変更の影響だけを取り出した検定になる。
#
# 変更前は `c50e17a`（`scale_init` を入れた 1a7a2f8 の1つ前）。
set -eo pipefail
source /opt/miniconda/3/etc/profile.d/conda.sh
cd "$(dirname "$0")/../.."
REPO=$(pwd)
conda activate sni-slam

BASE="${1:-c50e17a}"
OUT="$REPO/Registration/output/regress_scale_init"
WT=$(mktemp -d)/base
rm -rf "$OUT"; mkdir -p "$OUT"
cleanup() { git -C "$REPO" worktree remove --force "$WT" 2>/dev/null || true; }
trap cleanup EXIT

echo "### 入力を1回だけ作って固定する ###"
python -W ignore Registration/scripts/_regress_probe.py \
    --config Registration/configs/replica_room0.yaml --dump "$OUT/input.npz"

echo
echo "### 変更後（現在の作業ツリー） ###"
python -W ignore Registration/scripts/_regress_probe.py \
    --config Registration/configs/replica_room0.yaml \
    --load "$OUT/input.npz" --out "$OUT/after.json"

echo
echo "### 変更前（$BASE を worktree に取り出して実行） ###"
git -C "$REPO" worktree add --detach "$WT" "$BASE" >/dev/null
ln -s "$REPO/data" "$WT/data"
ln -s "$REPO/output" "$WT/output"
# 変更前のコミットには _regress_probe.py が無いので、現在のものを持ち込む。
# 呼ぶのは regbim 側の API だけなので、これで「変更前の regbim」を測れる。
cp Registration/scripts/_regress_probe.py "$WT/Registration/scripts/"
( cd "$WT" && python -W ignore Registration/scripts/_regress_probe.py \
    --config Registration/configs/replica_room0.yaml \
    --load "$OUT/input.npz" --out "$OUT/before.json" )

echo
echo "### 突き合わせ ###"
python - "$OUT" <<'PY'
import json, os, sys
import numpy as np
root = sys.argv[1]
a = json.load(open(os.path.join(root, "after.json")))
b = json.load(open(os.path.join(root, "before.json")))
bad = False
for key in sorted(set(a) | set(b)):
    if key not in a or key not in b:
        print("  %-22s ★片方にしか無い" % key); bad = True; continue
    va, vb = np.asarray(a[key], dtype=float), np.asarray(b[key], dtype=float)
    if va.shape != vb.shape:
        print("  %-22s ★形が違う" % key); bad = True; continue
    d = float(np.abs(va - vb).max())
    print("  %-22s 最大差 %.3e  %s" % (key, d, "一致" if d == 0.0 else "★不一致"))
    bad |= d != 0.0
print()
if bad:
    print("×  既定 median_axes の挙動が変わっている")
    raise SystemExit(1)
print("○  同じ入力に対して Sim(3) がビット単位で一致。既定の挙動は変わっていない")
PY
