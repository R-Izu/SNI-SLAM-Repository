#!/usr/bin/env bash
# IFC の下調べを bim-ifc env で実行する。
# sni-slam は py3.7 で ifcopenshell が入らないため env を分ける
# （BIM_IFC_Extraction/README.md に既にその運用が書かれている）。
set -eo pipefail
source /opt/miniconda/3/etc/profile.d/conda.sh
cd "$(dirname "$0")/../.."

IFC="${1:-BIM_IFC_Extraction/input/m3-411.ifc}"
OUT="${2:-Registration/output/ifc_survey.json}"
mkdir -p "$(dirname "$OUT")"

conda run -n bim-ifc python Registration/scripts/ifc_survey.py "$IFC" --out "$OUT" 2>&1 \
  | grep -v "No stream support"
