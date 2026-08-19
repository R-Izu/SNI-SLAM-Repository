#!/usr/bin/env bash
# 融合方式の比較を、**投影ラベル済みメッシュ**で公平に測り直す。
#
# 注意: TSDF のメッシュは写真の RGB 色を持つ。precheck_scene.py は頂点色を
# decode_segmap パレットとして解釈するので、生の TSDF メッシュに当てると
# ラベルが無意味になる。両方式とも project_labels_to_mesh.py を通した後の
# メッシュで比較しなければならない。
set -eo pipefail
source /opt/miniconda/3/etc/profile.d/conda.sh
cd "$(dirname "$0")/../.."
conda activate sni-slam

SCENE="${1:-m3_room_a}"

check () {  # <label> <projected-mesh> <out-json>
  echo "--- $1 ---"
  python -c "
import open3d as o3d
m = o3d.io.read_triangle_mesh('$2')
print('  頂点 %d / 三角 %d' % (len(m.vertices), len(m.triangles)))
"
  python tools/realdata/precheck_scene.py --mesh "$2" --scene "$1" --out "$3" 2>&1 \
      | grep -E "class fracs|gravity tilt|plane diversity"
  echo
}

check "SNI-SLAM融合(D)" \
      "output/RealData/_D/$SCENE/run1/mesh/final_mesh_semantic_projected.ply" \
      "output/RealData/_D/$SCENE/run1/precheck_projected.json"

check "TSDF融合" \
      "output/RealData/_TSDF/$SCENE/run1/mesh/final_mesh_semantic_projected.ply" \
      "output/RealData/_TSDF/$SCENE/run1/precheck_projected.json"
