#!/bin/bash
# R52 §3-2：実データ 4 スキャン × f = 0.2・0.4・0.6・0.8 の入力を作る（sni-slam の環境）。**回す前に commit する。**
# ① TSDF・② ラベルの投影とも、run_tsdf_bigscenes.sh と同じ引数（--voxel 0.04 --frame-stride 2、投影は --frame-stride 8）に、--frame-end-frac だけ足す。
# 出力：output/RealData/_TSDF/<scene>/r52_f<f>/mesh/final_mesh_semantic{,_projected}.ply と、それぞれの *_report.json
source /opt/miniconda/3/etc/profile.d/conda.sh && conda activate sni-slam
cd ~/rizu/SNI-SLAM
for s in m3_cor_a m3_cor_b m3_cor_c m3_cor_d; do
  for f in 0.2 0.4 0.6 0.8; do
    tag=$(echo $f | tr . p); o=output/RealData/_TSDF/$s/r52_f$tag/mesh
    [ -f $o/final_mesh_semantic_projected.ply ] && continue
    mkdir -p $o
    python -W ignore tools/realdata/tsdf_fusion.py --scene-dir data/realdata/$s --config configs/RealData/_D/$s.yaml \
        --out $o/final_mesh_semantic.ply --voxel 0.04 --frame-stride 2 --frame-end-frac $f 2>&1 | tail -1
    python -W ignore tools/realdata/project_labels_to_mesh.py --mesh $o/final_mesh_semantic.ply --scene-dir data/realdata/$s \
        --config configs/RealData/_D/$s.yaml --out $o/final_mesh_semantic_projected.ply --frame-stride 8 --frame-end-frac $f 2>&1 | tail -1
  done
done
echo inputs done
