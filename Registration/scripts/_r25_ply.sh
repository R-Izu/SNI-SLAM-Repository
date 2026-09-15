#!/bin/bash
# R25 追補：本人が目視するための点群を書き出す（--no-ply を外して回し直す）。
cd ~/rizu/SNI-SLAM
source /opt/miniconda/3/etc/profile.d/conda.sh
conda activate sni-slam
python Registration/scripts/export_stages.py --target m3_cor_c__E2 --scale-init vertical_only
python Registration/scripts/export_stages.py --target m3_cor_d__E3 --scale-init median_axes
python Registration/scripts/export_stages.py --target m3_cor_d__E3 --scale-init vertical_only
