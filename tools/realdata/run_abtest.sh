#!/usr/bin/env bash
# S2 A/B ゲート — 解像度案とラベル生成条件を、スキャン全体から等間隔サンプルした
# フレームで比較する。
#
# 比較する4条件
#   anam_fix   : anamorphic 格子 + 推論前に縦横比を戻す（本命）
#   anam_raw   : anamorphic 格子 + 引き伸ばしたまま推論（縦横比補正の効果を見る対照）
#   letterbox  : レターボックス格子 + 黒帯を background に固定
#   anam_norot : 回転補正なし（縦持ち補正の効果を見る対照）
#
# 先頭 N 枚だけだとカメラの向きが偏り天井が1枚も写らないことがあるため --sample を使う。
#
# -u は付けない: conda の activate.d が未設定変数を参照して落ちるため
set -eo pipefail

SCAN="${1:-b17452f252}"
NSAMP="${2:-24}"
OUTROOT=data/realdata_abtest

source /opt/miniconda/3/etc/profile.d/conda.sh
cd "$(dirname "$0")/../.."

rm -rf "$OUTROOT"

conda activate sni-slam
for m in anamorphic letterbox; do
  echo "=== convert: $m ==="
  python tools/realdata/convert_stray.py \
      --scan "$SCAN" --scene "_ab_$m" --mode "$m" \
      --sample "$NSAMP" --out "$OUTROOT" 2>&1 | tail -2
done

conda activate seg2d
echo "=== labels: anam_fix (aspect corrected) ==="
python tools/realdata/gen_labels_ade20k.py --scene-dir "$OUTROOT/_ab_anamorphic" \
    --out-name semantic_class --batch 4 2>&1 \
    | grep -E "^rotation|^aspect|^pixel fraction|^wall/floor|s/frame"

echo "=== labels: anam_raw (no aspect fix) ==="
python tools/realdata/gen_labels_ade20k.py --scene-dir "$OUTROOT/_ab_anamorphic" \
    --no-fix-aspect --out-name semantic_class_raw --batch 4 2>&1 \
    | grep -E "^rotation|^aspect|^pixel fraction|^wall/floor|s/frame"

echo "=== labels: letterbox ==="
python tools/realdata/gen_labels_ade20k.py --scene-dir "$OUTROOT/_ab_letterbox" \
    --out-name semantic_class --batch 4 2>&1 \
    | grep -E "^rotation|^aspect|^pixel fraction|^wall/floor|s/frame"

echo "=== labels: anam_norot (control) ==="
python tools/realdata/gen_labels_ade20k.py --scene-dir "$OUTROOT/_ab_anamorphic" \
    --no-rotate --out-name semantic_class_norot --batch 4 2>&1 \
    | grep -E "^rotation|^aspect|^pixel fraction|^wall/floor|s/frame"

conda activate sni-slam
python tools/realdata/compare_labels.py \
    --variant "anam_fix=$OUTROOT/_ab_anamorphic" \
    --variant "anam_raw=$OUTROOT/_ab_anamorphic:semantic_class_raw" \
    --variant "letterbox=$OUTROOT/_ab_letterbox" \
    --variant "anam_norot=$OUTROOT/_ab_anamorphic:semantic_class_norot" \
    --out output/RealData/_abtest --samples 6
