#!/bin/bash
# R48 段階1：GT-A の 8 参照（v2 の線分）の BIM 側の角・三つ組の数（r46_bim_count。登録はしない）
source /opt/miniconda/3/etc/profile.d/conda.sh && conda activate libim
export LiBIM_UST_ROOT=$HOME/rizu/r48_data/
STSL=$HOME/rizu/LiDAR2BIM-Registration
mkdir -p $STSL/configs/interval/r48
for d in $HOME/rizu/r48_data/BIM/GTA_*/; do
  f=$(basename $d)
  python - "$STSL/configs/interval/30m/2F/2f_office_02.yaml" "$STSL/configs/interval/r48/$f.yaml" "$f" <<'PY'
import sys, yaml
y = yaml.safe_load(open(sys.argv[1])); y["floor"] = sys.argv[3]; y["seq"] = "r48_count"
yaml.safe_dump(y, open(sys.argv[2], "w"), sort_keys=False)
PY
  $HOME/rizu/r46_stsl_build/r46_bim_count $STSL/configs/interval/r48/$f.yaml 2>/dev/null | grep "^{"
  rm -f $d/bimHash*.bin
done
