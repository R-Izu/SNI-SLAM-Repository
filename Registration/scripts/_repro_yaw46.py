"""46 件の1つを再現し、候補の回転誤差と最終解の回転誤差の食い違いを特定する。

観測：勝者の回転誤差 180.0°、最良候補でも 90°、**4候補のスコアが全て同一**。
それでも `success=True` になっていた。**成功判定は最終解の回転誤差 < 5° を要求する**ので、
どこかで「候補の回転」と「最終解の回転」が別物になっている。

対象：`office_0__cov030_ne__median_axes` の trial 0。
"""

from __future__ import annotations

import copy
import json
import os
import sys

import numpy as np
import yaml

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "Registration"))
os.chdir(REPO)

from regbim import io_utils, metrics, preprocess          # noqa: E402
from regbim.config import load_t_gt                       # noqa: E402
from regbim.methods import get_method                     # noqa: E402
from regbim.metrics import rotation_error_deg             # noqa: E402

cfg = yaml.safe_load(open("Registration/configs/sectionC/office_0.yaml"))
cfg["reference"]["clip"] = {"keep_frac": 0.30, "anchor": [1.0, 1.0]}   # ne
cfg.setdefault("proposed", {})["scale_init"] = "median_axes"
cfg.setdefault("diagnostics", {})["record_yaw"] = True

src = io_utils.load_source_cloud(cfg)
dst = io_utils.load_reference_cloud(cfg)
m = get_method("proposed")

T0 = m.register(src, dst, cfg)
print("無摂動解 T0 の scale = %.4f" % metrics.decompose_sim3(T0)[2])

rng = np.random.default_rng(int(cfg["eval"]["seed"]))
P = metrics.random_sim3(rng, cfg["eval"]["perturb"])      # trial 0 の摂動
src_p = metrics.transform_cloud(src, P)
Tt = m.register(src_p, dst, cfg)
d = m.last_yaw_diag

expected = T0 @ metrics.invert_sim3(P)
e = metrics.sim3_errors(Tt, expected)
R_exp = metrics.decompose_sim3(expected)[0]
R_out = metrics.decompose_sim3(Tt)[0]

print()
print("候補スコア            : %s" % d["candidate_scores"])
print("勝者 index            : %d" % d["winner"])
errs = [rotation_error_deg(np.asarray(R), R_exp) for R in d["candidate_R"]]
print("候補ごとの回転誤差[度]: %s" % [round(x, 3) for x in errs])
print("argmin(候補)          : %d" % int(np.argmin(errs)))
print()
print("**最終解の回転誤差**  : %.3f 度   （sim3_errors）" % e["rot_deg"])
print("勝者候補の回転誤差    : %.3f 度" % errs[d["winner"]])
print("最終解の回転 と 勝者候補の回転 の差: %.3f 度"
      % rotation_error_deg(R_out, np.asarray(d["candidate_R"][d["winner"]])))
print()
print("並進誤差 %.4f m / 縮尺比 %.4f" % (e["trans"], e["scale_ratio"]))
succ = cfg["eval"]["success"]
ok = (e["rot_deg"] < succ["rot_deg"] and e["trans"] < succ["trans"]
      and e["scale_ratio"] < succ["scale_ratio"])
print("成功判定（自己一貫性）: %s" % ok)
