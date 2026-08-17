#!/usr/bin/env bash
# S1 疎通確認 — b17452f252（唯一ループが閉じている／床 α-shape 最大）を 1 run 通す。
#
# 併せて指示書追補の T-A3 用に、depth 制限あり／なしの2条件を作る。
# RGB と意味ラベルは2条件で同一なので、depth だけ別に作って rgb/semantic は共有する
# （RGB 8452 枚の再エンコードと推論をもう一度やらずに済む）。
#
# 各ステップは成果物があれば飛ばす（再実行可能）。変換 50 分・ラベル 16 分を
# 失敗のたびにやり直さないため。
#
# -u は付けない: conda の activate.d が未設定変数を参照して落ちるため
set -eo pipefail

SCAN="${1:-b17452f252}"
SCENE="${2:-m3_cor_d}"
BATCH="${3:-8}"

source /opt/miniconda/3/etc/profile.d/conda.sh
cd "$(dirname "$0")/../.."
ROOT="$PWD"
LOG="$ROOT/output/RealData/_logs"
CTRL="${SCENE}_nodepthlimit"
mkdir -p "$LOG"

step() { echo; echo "########## $* ##########"; date -Is; }
have()  { [ -e "$1" ]; }
nfiles() { ls "$1" 2>/dev/null | wc -l; }

# ---------------------------------------------------------------- 1. 変換（採用条件）
step "1/6 convert $SCAN -> $SCENE (anamorphic, conf>=1, depth<=5m)"
conda activate sni-slam
if have "data/realdata/$SCENE/conversion_report.json"; then
  echo "skip: already converted ($(nfiles data/realdata/$SCENE/rgb) rgb frames)"
else
  python tools/realdata/convert_stray.py \
      --scan "$SCAN" --scene "$SCENE" --mode anamorphic \
      --conf-min 1 --max-depth 5.0 --out data/realdata
fi

# ---------------------------------------------------------------- 2. 意味ラベル
step "2/6 semantic labels (ADE20K, rotation corrected)"
if have "data/realdata/$SCENE/label_report_semantic_class.json"; then
  echo "skip: already labelled ($(nfiles data/realdata/$SCENE/semantic_class) frames)"
else
  conda activate seg2d
  python tools/realdata/gen_labels_ade20k.py \
      --scene-dir "data/realdata/$SCENE" --batch "$BATCH"
  conda activate sni-slam
fi

# ---------------------------------------------------------------- 3. 契約チェック
step "3/6 validate_scene_data"
conda activate sni-slam
python Registration/scripts/validate_scene_data.py "data/realdata/$SCENE" \
    --out "$LOG/validate_$SCENE.json"

# ---------------------------------------------------------------- 4. T-A3 の対照条件
# depth 無制限版。rgb/ semantic_class/ は採用条件と共有（シンボリックリンク）。
step "4/6 build T-A3 control scene (no depth limit)"
if have "data/realdata/$CTRL/conversion_report.json"; then
  echo "skip: control already built"
else
  python tools/realdata/convert_stray.py \
      --scan "$SCAN" --scene "$CTRL" --mode anamorphic \
      --conf-min 1 --max-depth 0 --out data/realdata \
      --depth-only "data/realdata/$SCENE"
fi
ln -sfn "$ROOT/data/realdata/$SCENE/rgb"            "data/realdata/$CTRL/rgb"
ln -sfn "$ROOT/data/realdata/$SCENE/semantic_class" "data/realdata/$CTRL/semantic_class"
python Registration/scripts/validate_scene_data.py "data/realdata/$CTRL" \
    --out "$LOG/validate_$CTRL.json"

# ---------------------------------------------------------------- 5. SLAM 本番
for s in "$SCENE" "$CTRL"; do
  step "5/6 SLAM run: $s"
  if have "output/RealData/$s/run1/mesh/final_mesh_semantic.ply"; then
    echo "skip: mesh already exists"
  else
    python -W ignore run.py "configs/RealData/$s.yaml" \
        --output "output/RealData/$s/run1" 2>&1 | tee "$LOG/slam_${s}_run1.log"
  fi
done

# ---------------------------------------------------------------- 6. 評価と後片付け
step "6/6 eval_ate + precheck + feat_cache cleanup"
for s in "$SCENE" "$CTRL"; do
  python src/tools/eval_ate.py "configs/RealData/$s.yaml" \
      --output "output/RealData/$s/run1" 2>&1 | tail -20 || echo "eval_ate failed for $s"
  python tools/realdata/precheck_scene.py \
      --mesh "output/RealData/$s/run1/mesh/final_mesh_semantic.ply" \
      --traj-gt "data/realdata/$SCENE/traj.txt" \
      --est-poses "output/RealData/$s/run1/ckpts" \
      --scene "$s" --out "output/RealData/$s/run1/precheck_$s.json" \
      || echo "precheck failed for $s"
  rm -rf "output/RealData/$s/run1/feat_cache"
done

step "S1 done"
du -sh output/RealData/*/run1 2>/dev/null || true
