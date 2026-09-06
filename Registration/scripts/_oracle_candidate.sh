#!/usr/bin/env bash
# 「正解を候補集合に入れたら選ばれるか」を確かめる。
#
# 撤回（2026-09-07）で「目的関数の順位付けが壊れている」証拠が無くなったので、
# 残る仮説は「探索が正解に到達していない」である。
# **正解 G を候補に混ぜて通常どおり ICP＋採点すれば、1回で決まる。**
#   選ばれる   -> 採点は正しい。**探索の問題**
#   選ばれない -> 採点の問題が残っている
set -eo pipefail
source /opt/miniconda/3/etc/profile.d/conda.sh
cd /home/student/rizu/SNI-SLAM
conda activate sni-slam

python -W ignore - <<'PY'
import json, sys
import numpy as np
import yaml
sys.path.insert(0, "Registration")
from regbim import io_utils, preprocess, metrics
from regbim.metrics import class_inlier_ratio
from regbim.semantic_icp import semantic_icp
from regbim.methods import get_method

for scene, cond in (("m3_cor_c", "E2"), ("m3_cor_a", "E3")):
    p = "Registration/configs/realdata/%s__%s.yaml" % (scene, cond)
    cfg = yaml.safe_load(open(p))
    cfg["source"]["seed"] = 0
    cfg["reference"]["seed"] = 0
    thr = cfg["semantic_icp"]["max_corr_dist"]
    src = io_utils.load_source_cloud(cfg)
    dst = io_utils.load_reference_cloud(cfg)
    sp = preprocess.prepare(src, cfg)
    dp = preprocess.prepare(dst, cfg)
    G = np.array(json.load(open(cfg["eval"]["t_gt_path"]))["T_gt"])

    T_method = get_method("proposed").register(src, dst, cfg)
    # 正解を「候補」として同じ ICP に通す（他の候補と同じ扱いにする）
    T_from_G = semantic_icp(sp, dp, G, cfg, rotation_fixed=True)

    rG = class_inlier_ratio(sp, dp, G, thr)
    rM = class_inlier_ratio(sp, dp, T_method, thr)
    rGi = class_inlier_ratio(sp, dp, T_from_G, thr)
    eM = metrics.sim3_errors(T_method, G)
    eGi = metrics.sim3_errors(T_from_G, G)
    print("=== %s %s ===" % (scene, cond))
    print("  正解 G そのもの          rho = %.5f" % rG)
    print("  手法が返した姿勢          rho = %.5f  回転 %.2f度 並進 %.3f m"
          % (rM, eM["rot_deg"], eM["trans"]))
    print("  G を初期値に ICP した結果 rho = %.5f  回転 %.2f度 並進 %.3f m"
          % (rGi, eGi["rot_deg"], eGi["trans"]))
    win = "G 由来" if rGi > rM else "手法の姿勢"
    print("  -> 同じスコアで比べると **%s** が勝つ" % win)
    if rGi > rM:
        print("     ＝ 採点は正解を選べる。**探索が到達していない**のが問題")
    else:
        print("     ＝ 採点が正解を選べない。**採点側の問題が残る**")
PY
