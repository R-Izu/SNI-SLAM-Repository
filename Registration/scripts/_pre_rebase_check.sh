#!/usr/bin/env bash
# rebase 前のコミット（50c34f2）で、報告した 0.339 が再現するかを確かめる。
# 再現しなければ、報告した値が何に由来したのかを別途特定する必要がある。
set -eo pipefail
source /opt/miniconda/3/etc/profile.d/conda.sh
cd /home/student/rizu/SNI-SLAM
conda activate sni-slam

WT=/tmp/pre_rebase
git worktree remove --force "$WT" 2>/dev/null || true
rm -rf "$WT"
git worktree add --detach "$WT" 50c34f2 >/dev/null
ln -sfn /home/student/rizu/SNI-SLAM/data "$WT/data"
ln -sfn /home/student/rizu/SNI-SLAM/output "$WT/output"
# IFC キャッシュは Registration/output/ 側にある
rm -rf "$WT/Registration/output"
ln -sfn /home/student/rizu/SNI-SLAM/Registration/output "$WT/Registration/output"

cd "$WT"
python -W ignore - <<'PY'
import copy, json, yaml, sys
import numpy as np
sys.path.insert(0, "Registration")
from regbim import io_utils, preprocess, metrics
from regbim.methods import get_method
from regbim.metrics import class_inlier_ratio

p = "Registration/configs/realdata/m3_cor_c__E2.yaml"
cfg = yaml.safe_load(open(p))
thr = cfg["semantic_icp"]["max_corr_dist"]
src = io_utils.load_source_cloud(cfg)
dst = io_utils.load_reference_cloud(cfg)
sp = preprocess.prepare(src, cfg)
dp = preprocess.prepare(dst, cfg)
G = np.array(json.load(open(cfg["eval"]["t_gt_path"]))["T_gt"])
print("rebase 前（50c34f2）で再測定")
print("  正解 rho = %.5f" % class_inlier_ratio(sp, dp, G, thr))
for mode in ("centroid", "plane_match"):
    c = copy.deepcopy(cfg)
    c.setdefault("proposed", {})["translation_init"] = mode
    T = get_method("proposed").register(src, dst, c)
    e = metrics.sim3_errors(T, G)
    print("  %-12s rho = %.5f  回転 %.2f度  並進 %.3f m  s = %.4f"
          % (mode, class_inlier_ratio(sp, dp, T, thr), e["rot_deg"], e["trans"],
             metrics.decompose_sim3(T)[2]))
PY

cd /home/student/rizu/SNI-SLAM
git worktree remove --force "$WT" 2>/dev/null || true
