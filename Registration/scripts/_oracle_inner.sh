#!/usr/bin/env bash
# R9 §3-3-1 — 参照を**室内面のみ**に絞って、オラクル実験をやり直す。
#
# 前回：正しい初期値（基準変換 G）から ICP を回しても、収束先が G から
#       0.175 m / 0.293 m ずれた。**成功閾値 0.1 m を超える。**
# 仮説：GT の精緻化は室内面のみで行い、手法は表裏両面に合わせている。
#       壁厚 0.150〜0.200 m はゲート 0.3 m の内側なので、手法は壁の中心へ引かれる。
# 判定：室内面に絞って残差が減れば仮説は支持される。減らなければ落ちる。
set -eo pipefail
source /opt/miniconda/3/etc/profile.d/conda.sh
cd /home/student/rizu/SNI-SLAM
conda activate sni-slam

python -W ignore - <<'PY'
import copy, json, sys
import numpy as np
import yaml
sys.path.insert(0, "Registration")
from regbim import io_utils, preprocess, metrics
from regbim.metrics import class_inlier_ratio
from regbim.semantic_icp import semantic_icp
from regbim.methods import get_method

print("%-14s %-12s %9s %10s %11s %9s"
      % ("scene/cond", "参照", "参照点数", "rho", "並進誤差m", "回転度"))
print("-" * 70)
for scene, cond in (("m3_cor_c", "E2"), ("m3_cor_a", "E3")):
    base = yaml.safe_load(open("Registration/configs/realdata/%s__%s.yaml"
                               % (scene, cond)))
    base["source"]["seed"] = 0
    base["reference"]["seed"] = 0
    thr = base["semantic_icp"]["max_corr_dist"]
    G = np.array(json.load(open(base["eval"]["t_gt_path"]))["T_gt"])
    src = io_utils.load_source_cloud(base)
    sp = preprocess.prepare(src, base)
    for inner in (False, True):
        cfg = copy.deepcopy(base)
        cfg["reference"]["inner_only"] = inner
        dst = io_utils.load_reference_cloud(cfg)
        dp = preprocess.prepare(dst, cfg)
        # 正解を初期値にした ICP（＝オラクル）
        T = semantic_icp(sp, dp, G, cfg, rotation_fixed=True)
        e = metrics.sim3_errors(T, G)
        print("%-14s %-12s %9d %10.5f %11.3f %9.2f"
              % ("%s %s" % (scene, cond), "室内面のみ" if inner else "表裏両面",
                 len(dst), class_inlier_ratio(sp, dp, T, thr),
                 e["trans"], e["rot_deg"]))
PY
