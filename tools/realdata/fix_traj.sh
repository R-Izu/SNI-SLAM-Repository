#!/usr/bin/env bash
# 変換済み全シーンの traj.txt を、正しい姿勢規約（素の OpenCV c2w）で書き直す。
# rgb/depth/semantic は一切触らないので数秒で終わる。
#
# 背景: 当初「ローダが col 1,2 を反転するので先に反転して打ち消す」と実装していたが、
# ローダの反転は OpenCV -> OpenGL 変換そのものであり、打ち消してはいけなかった。
# 二重反転で SLAM から見たカメラの y,z 軸が逆になり、再構成が壊れていた。
set -eo pipefail
source /opt/miniconda/3/etc/profile.d/conda.sh
cd "$(dirname "$0")/../.."
conda activate sni-slam

for d in data/realdata/*/; do
  s=$(basename "$d")
  [ -f "$d/conversion_report.json" ] || { echo "skip $s (not converted)"; continue; }
  python tools/realdata/convert_stray.py --scene "$s" --scan dummy --traj-only --out data/realdata
done

echo
echo "=== 検証: 各シーンの最下面が水平か ==="
python tools/realdata/verify_traj.py
