#!/usr/bin/env bash
# R3 T2 の受入条件 —「`scale_init: median_axes`（既定）で既存結果が完全に再現する」
#
# 構造的には自明（既定では使う軸の集合が全軸のままなので `seed_scale` は同じ式を通る）が、
# **自明だから測らない、はやらない。** HEAD のコードを同じ seed で走らせ、
# 試行ごとの誤差まで一致するかを突き合わせる。
#
# ★作業ツリーの proposed.py を書き換えてはならない。
#   最初の版はそれをやって、並行して編集していた変更を巻き添えで消した。
#   HEAD 側は **git worktree で別ディレクトリに取り出して**走らせる。
#   データと出力は重い＆共有なので、worktree からは元repoへシンボリックリンクを張る。
set -eo pipefail
source /opt/miniconda/3/etc/profile.d/conda.sh
cd "$(dirname "$0")/../.."
REPO=$(pwd)
conda activate sni-slam

N="${1:-20}"
OUT="$REPO/Registration/output/regress_scale_init"
WT=$(mktemp -d)/head
rm -rf "$OUT"; mkdir -p "$OUT"
cleanup() { git -C "$REPO" worktree remove --force "$WT" 2>/dev/null || true; }
trap cleanup EXIT

echo "### 変更後（現在の作業ツリー） ###"
python -W ignore Registration/scripts/benchmark.py \
    --config Registration/configs/replica_room0.yaml \
    --methods proposed --trials "$N" --out-dir "$OUT/after" 2>&1 | tail -2

echo
echo "### 変更前（git HEAD を worktree に取り出して実行） ###"
git -C "$REPO" worktree add --detach "$WT" HEAD >/dev/null
ln -s "$REPO/data" "$WT/data"
ln -s "$REPO/output" "$WT/output"
ln -s "$REPO/Registration/output" "$WT/Registration/output_link" 2>/dev/null || true
mkdir -p "$WT/Registration/output"
cp "$REPO/Registration/output/eval/T_gt.json" "$WT/Registration/output/eval/T_gt.json" 2>/dev/null \
  || { mkdir -p "$WT/Registration/output/eval"; \
       cp "$REPO/Registration/output/eval/T_gt.json" "$WT/Registration/output/eval/"; }
( cd "$WT" && python -W ignore Registration/scripts/benchmark.py \
    --config Registration/configs/replica_room0.yaml \
    --methods proposed --trials "$N" --out-dir "$OUT/before" 2>&1 | tail -2 )

echo
echo "### 突き合わせ ###"
python - "$OUT" <<'PY'
import glob, json, os, sys
root = sys.argv[1]
def load(d):
    out = {}
    for p in sorted(glob.glob(os.path.join(d, "**", "*"), recursive=True)):
        if not os.path.isfile(p) or os.path.splitext(p)[1] not in (".json", ".csv"):
            continue
        with open(p) as f:
            out[os.path.relpath(p, d)] = f.read()
    return out
a, b = load(os.path.join(root, "after")), load(os.path.join(root, "before"))
if set(a) != set(b):
    print("×  出力ファイルの集合が違う\n  after : %s\n  before: %s"
          % (sorted(a), sorted(b)))
    raise SystemExit(1)

# 所要時間・パス・commit は一致しなくて当然なので、そこだけ落として比べる
DROP = ("time", "out_dir", "config", "git_commit", "timestamp")
def strip(o):
    if isinstance(o, dict):
        return {k: strip(v) for k, v in o.items() if not any(d in k for d in DROP)}
    if isinstance(o, list):
        return [strip(v) for v in o]
    return o
def norm(name, text):
    if name.endswith(".json"):
        return json.dumps(strip(json.loads(text)), sort_keys=True)
    rows = [r.split(",") for r in text.strip().splitlines()]
    hdr = rows[0]
    keep = [i for i, h in enumerate(hdr) if not any(d in h for d in DROP)]
    return json.dumps([[r[i] for i in keep] for r in rows])

bad = [k for k in a if norm(k, a[k]) != norm(k, b[k])]
for k in sorted(a):
    print("  %-24s %s" % (k, "一致" if k not in bad else "★不一致"))
print()
if bad:
    print("×  既定の挙動が変わっている: %s" % bad)
    raise SystemExit(1)
print("○  所要時間・パス以外のすべての値が一致。既定 median_axes の挙動は変わっていない")
PY
