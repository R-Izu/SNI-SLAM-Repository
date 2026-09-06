#!/usr/bin/env bash
# R9 §2-2-2 — 撤回した 0.34 が「裾」なのかを見る。
# 0.339 を出したのと同じコミット 50c34f2 で、seed を 20 本振って rho(W) の分布を出す。
set -eo pipefail
source /opt/miniconda/3/etc/profile.d/conda.sh
cd /home/student/rizu/SNI-SLAM
conda activate sni-slam

WT=/tmp/seed20
git worktree remove --force "$WT" 2>/dev/null || true
rm -rf "$WT"
git worktree add --detach "$WT" 50c34f2 >/dev/null
ln -sfn /home/student/rizu/SNI-SLAM/data "$WT/data"
ln -sfn /home/student/rizu/SNI-SLAM/output "$WT/output"
rm -rf "$WT/Registration/output"
ln -sfn /home/student/rizu/SNI-SLAM/Registration/output "$WT/Registration/output"

cd "$WT"
python -W ignore - <<'PY'
import copy, json, sys
import numpy as np
import yaml
sys.path.insert(0, "Registration")
from regbim import io_utils, preprocess, metrics
from regbim.metrics import class_inlier_ratio
from regbim.methods import get_method

p = "Registration/configs/realdata/m3_cor_c__E2.yaml"
base = yaml.safe_load(open(p))
thr = base["semantic_icp"]["max_corr_dist"]
G = np.array(json.load(open(base["eval"]["t_gt_path"]))["T_gt"])
rows = []
print("%-5s %9s %9s %9s %9s" % ("seed", "rho_G", "rho_W", "回転度", "s_W"))
for seed in range(20):
    cfg = copy.deepcopy(base)
    cfg["source"]["seed"] = seed
    cfg["reference"]["seed"] = seed
    src = io_utils.load_source_cloud(cfg)
    dst = io_utils.load_reference_cloud(cfg)
    sp = preprocess.prepare(src, cfg)
    dp = preprocess.prepare(dst, cfg)
    c = copy.deepcopy(cfg)
    c.setdefault("proposed", {})["translation_init"] = "plane_match"
    T = get_method("proposed").register(src, dst, c)
    rG = class_inlier_ratio(sp, dp, G, thr)
    rW = class_inlier_ratio(sp, dp, T, thr)
    e = metrics.sim3_errors(T, G)
    rows.append({"seed": seed, "rho_G": rG, "rho_W": rW,
                 "rot_deg": e["rot_deg"], "s": float(metrics.decompose_sim3(T)[2])})
    print("%-5d %9.5f %9.5f %9.2f %9.4f" % (seed, rG, rW, e["rot_deg"], rows[-1]["s"]))
w = np.array([r["rho_W"] for r in rows])
g = np.array([r["rho_G"] for r in rows])
print("\nrho_W: 中央値 %.5f / 最小 %.5f / 最大 %.5f / 標準偏差 %.5f"
      % (np.median(w), w.min(), w.max(), w.std()))
print("rho_G: 中央値 %.5f / 最小 %.5f / 最大 %.5f" % (np.median(g), g.min(), g.max()))
print("rho_W > rho_G となった回数: %d / %d" % (int((w > g).sum()), len(w)))
print("rho_W >= 0.30 となった回数: %d / %d" % (int((w >= 0.30).sum()), len(w)))
json.dump(rows, open("/home/student/rizu/SNI-SLAM/Registration/output/diag/seed20.json", "w"),
          indent=2)
PY

cd /home/student/rizu/SNI-SLAM
git worktree remove --force "$WT" 2>/dev/null || true
