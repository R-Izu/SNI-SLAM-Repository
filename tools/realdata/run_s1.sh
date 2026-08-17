#!/usr/bin/env bash
# S1 疎通確認 — b17452f252（唯一ループが閉じている／床 α-shape 最大）を 1 run 通す。
#
# 併せて指示書追補の T-A3 用に、depth 制限あり／なしの2条件を作る。
# RGB と意味ラベルは2条件で同一なので、depth だけ別に作って rgb/semantic は共有する
# （RGB 8452 枚の再エンコードと 70 分の推論をもう一度やらずに済む）。
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
mkdir -p "$LOG"

step() { echo; echo "########## $* ##########"; date -Is; }

# ---------------------------------------------------------------- 1. 変換（採用条件）
step "1/6 convert $SCAN -> $SCENE (anamorphic, conf>=1, depth<=5m)"
conda activate sni-slam
python tools/realdata/convert_stray.py \
    --scan "$SCAN" --scene "$SCENE" --mode anamorphic \
    --conf-min 1 --max-depth 5.0 --out data/realdata

# ---------------------------------------------------------------- 2. 意味ラベル
step "2/6 semantic labels (ADE20K, rotation corrected)"
conda activate seg2d
python tools/realdata/gen_labels_ade20k.py \
    --scene-dir "data/realdata/$SCENE" --batch "$BATCH"

# ---------------------------------------------------------------- 3. 契約チェック
step "3/6 validate_scene_data"
conda activate sni-slam
python Registration/scripts/validate_scene_data.py "data/realdata/$SCENE" \
    --out "$LOG/validate_$SCENE.json"

# ---------------------------------------------------------------- 4. T-A3 の対照条件
# depth 無制限版。rgb/ semantic_class/ traj.txt は採用条件と共有（シンボリックリンク）。
step "4/6 build T-A3 control scene (no depth limit)"
CTRL="${SCENE}_nodepthlimit"
python tools/realdata/convert_stray.py \
    --scan "$SCAN" --scene "$CTRL" --mode anamorphic \
    --conf-min 1 --max-depth 0 --out data/realdata \
    --depth-only "data/realdata/$SCENE"
ln -sfn "$ROOT/data/realdata/$SCENE/rgb"            "data/realdata/$CTRL/rgb"
ln -sfn "$ROOT/data/realdata/$SCENE/semantic_class" "data/realdata/$CTRL/semantic_class"
python Registration/scripts/validate_scene_data.py "data/realdata/$CTRL" \
    --out "$LOG/validate_$CTRL.json"

# ---------------------------------------------------------------- 5. SLAM 本番
step "5/6 SLAM run: $SCENE"
python -W ignore run.py "configs/RealData/$SCENE.yaml" \
    --output "output/RealData/$SCENE/run1" 2>&1 | tee "$LOG/slam_${SCENE}_run1.log"

step "5b/6 SLAM run: $CTRL (T-A3 control)"
python -W ignore run.py "configs/RealData/$CTRL.yaml" \
    --output "output/RealData/$CTRL/run1" 2>&1 | tee "$LOG/slam_${CTRL}_run1.log"

# ---------------------------------------------------------------- 6. 評価と後片付け
step "6/6 eval_ate + feat_cache cleanup"
for s in "$SCENE" "$CTRL"; do
  python src/tools/eval_ate.py "configs/RealData/$s.yaml" \
      --output "output/RealData/$s/run1" 2>&1 | tail -20 || echo "eval_ate failed for $s"
  rm -rf "output/RealData/$s/run1/feat_cache"
done

step "S1 done"
du -sh output/RealData/*/run1 2>/dev/null || true
