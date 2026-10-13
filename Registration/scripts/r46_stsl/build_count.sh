#!/bin/bash
# R46：r46_bim_count を、STSL の reg_bm と同じコンパイル・リンク設定でビルドする（STSL のリポジトリは変えない）
set -e
source /opt/miniconda/3/etc/profile.d/conda.sh && conda activate libim
STSL=$HOME/rizu/LiDAR2BIM-Registration
SRC=$HOME/rizu/SNI-SLAM/Registration/scripts/r46_stsl/r46_bim_count.cpp
OUT=$HOME/rizu/r46_stsl_build
cd $STSL/build
FL=$(grep -h "^CXX_FLAGS\|^CXX_DEFINES\|^CXX_INCLUDES" CMakeFiles/reg_bm.dir/flags.make | sed 's/^[A-Z_]* = //' | tr '\n' ' ')
eval c++ $FL -c $SRC -o $OUT/r46_bim_count.o
LINK=$(cat CMakeFiles/reg_bm.dir/link.txt)
LINK=${LINK//CMakeFiles\/reg_bm.dir\/examples\/cpp\/reg_bm.cpp.o/$OUT/r46_bim_count.o}
LINK=${LINK//$STSL\/bin\/reg_bm/$OUT/r46_bim_count}
LINK=$(echo "$LINK" | sed 's#-Wl,--dependency-file=[^ ]*##')
eval $LINK
ls -la $OUT/r46_bim_count
