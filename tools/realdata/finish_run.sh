#!/usr/bin/env bash
# 完走した run に対して、評価と事前チェックと後片付けだけを行う。
# run.py を直接叩いたとき（run_s1.sh / run_batch.py を通していないとき）に使う。
#
# 使い方: finish_run.sh <scene> [run_no] [traj_scene]
#   traj_scene は traj.txt の出どころ。T-A3 の対照シーンのように
#   rgb/semantic を別シーンと共有している場合に指定する。
set -eo pipefail
source /opt/miniconda/3/etc/profile.d/conda.sh
cd "$(dirname "$0")/../.."
conda activate sni-slam

SCENE="${1:?scene name required}"
RUN="${2:-1}"
TRAJ_SCENE="${3:-$SCENE}"
OUT="output/RealData/${SCENE}/run${RUN}"

echo "=== eval_ate: ${SCENE} run${RUN} ==="
python src/tools/eval_ate.py "configs/RealData/${SCENE}.yaml" --output "$OUT" 2>&1 | tail -20 \
  || echo "eval_ate failed (ARKit 姿勢を疑似GTにした参考値なので、失敗しても致命的ではない)"

echo
echo "=== precheck_scene: ${SCENE} ==="
python tools/realdata/precheck_scene.py \
    --mesh "${OUT}/mesh/final_mesh_semantic.ply" \
    --traj-gt "data/realdata/${TRAJ_SCENE}/traj.txt" \
    --est-poses "${OUT}/ckpts" \
    --scene "$SCENE" --out "${OUT}/precheck_${SCENE}.json"

echo
echo "=== cleanup feat_cache ==="
du -sh "${OUT}/feat_cache" 2>/dev/null || true
rm -rf "${OUT}/feat_cache"
du -sh "$OUT"
