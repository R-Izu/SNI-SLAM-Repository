#!/bin/bash
# R46：STSL の Thirdparty/point-sam/examples/plane_seg.cpp の process() をそのまま写し、main だけを
# 「引数の config 1 つを処理する」に替えた版を作る（公式の main は config の一覧が直書きで、引数を読まない）。
# STSL のリポジトリは変えない。生成したソースは r46_plane_seg.cpp として SNI-SLAM 側に残す。
set -e
source /opt/miniconda/3/etc/profile.d/conda.sh && conda activate libim
STSL=$HOME/rizu/LiDAR2BIM-Registration
HERE=$HOME/rizu/SNI-SLAM/Registration/scripts/r46_stsl
OUT=$HOME/rizu/r46_stsl_build
SRC=$STSL/Thirdparty/point-sam/examples/plane_seg.cpp
N=$(grep -n "^int main" $SRC | cut -d: -f1)
{ echo "// 自動生成：STSL $(cd $STSL && git rev-parse --short HEAD) の point-sam/examples/plane_seg.cpp の 1〜$((N-1)) 行目をそのまま写し、main を替えた"
  head -n $((N-1)) $SRC
  echo 'int main(int argc, char **argv) {'
  echo '    if (argc < 2) { std::cerr << "usage: r46_plane_seg <pointsam config>" << std::endl; return 2; }'
  echo '    process(argv[1], true, true);'
  echo '    return 0;'
  echo '}'
} > $HERE/r46_plane_seg.cpp
cd $STSL/Thirdparty/point-sam/build
FL=$(grep -h "^CXX_FLAGS\|^CXX_DEFINES\|^CXX_INCLUDES" CMakeFiles/plane_seg.dir/flags.make | sed 's/^[A-Z_]* = //' | tr '\n' ' ')
eval c++ $FL -c $HERE/r46_plane_seg.cpp -o $OUT/r46_plane_seg.o
LINK=$(cat CMakeFiles/plane_seg.dir/link.txt)
LINK=${LINK//CMakeFiles\/plane_seg.dir\/examples\/plane_seg.cpp.o/$OUT/r46_plane_seg.o}
LINK=$(echo "$LINK" | sed "s# -o plane_seg # -o $OUT/r46_plane_seg #; s#-Wl,--dependency-file=[^ ]*##")
eval $LINK
ls -la $OUT/r46_plane_seg
