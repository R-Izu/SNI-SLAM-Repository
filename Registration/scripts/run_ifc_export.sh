#!/usr/bin/env bash
# R3 T1 受入条件 — `["411"]` / `["410"]` / `["411","410"]` の3通り＋全体を出す。
# ifcopenshell は bim-ifc env にしか無い（sni-slam は py3.7）。
set -eo pipefail
source /opt/miniconda/3/etc/profile.d/conda.sh
cd "$(dirname "$0")/../.."
conda activate bim-ifc

IFC=BIM_IFC_Extraction/input/m3-411.ifc
CFG=Registration/configs/m3_ifc.yaml
OUT=Registration/output/ifc
mkdir -p "$OUT"

echo "########## 全体（切り出しなし・構造クラスのみ） ##########"
python Registration/scripts/ifc_export.py --ifc "$IFC" --config "$CFG" \
    --out "$OUT/m3_ifc_all.npz" --ply

for s in 411 410; do
  echo
  echo "########## spaces=[$s] ##########"
  python Registration/scripts/ifc_export.py --ifc "$IFC" --config "$CFG" \
      --spaces "$s" --out "$OUT/m3_ifc_$s.npz" --ply 2>&1 | grep -vE "^  id=|^IfcSpace:"
done

echo
echo "########## spaces=[411,410] ##########"
python Registration/scripts/ifc_export.py --ifc "$IFC" --config "$CFG" \
    --spaces 411 410 --out "$OUT/m3_ifc_411_410.npz" --ply 2>&1 | grep -vE "^  id=|^IfcSpace:"

echo
echo "########## 診断: keep_classes を外して全クラスの内訳を見る ##########"
python Registration/scripts/ifc_export.py --ifc "$IFC" --config "$CFG" \
    --no-keep-classes --out "$OUT/m3_ifc_allclasses.npz" 2>&1 | grep -E "要素数|点数"
