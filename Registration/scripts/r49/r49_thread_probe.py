"""R49 段階0：GT-A room_0 の試行 0 の T_est が、スレッドの数でビット一致しなくなるかを見る（1 試行を 1 回）。"""
import json, os, sys
import numpy as np
import yaml

REPO = os.path.expanduser("~/rizu/SNI-SLAM")
sys.path.insert(0, os.path.join(REPO, "Registration")); os.chdir(REPO)
from regbim import io_utils, metrics                             # noqa: E402
from regbim.methods import get_method                            # noqa: E402

cfg = yaml.safe_load(open("Registration/configs/gt_a_plan_a/room_0.yaml"))
tm = json.load(open("output/Registration/r45/plan_a/room_0/trial_matrices.json"))
row = [t for t in tm["trials"] if t["trial"] == int(sys.argv[1])][0]
src = io_utils.load_source_cloud(cfg); dst = io_utils.load_reference_cloud(cfg)
T = np.asarray(get_method("proposed").register(metrics.transform_cloud(src, np.asarray(row["P"])), dst, cfg))
R0 = np.asarray(row["T_est"])
d = np.abs(T - R0).max()
Ra, ta, sa = metrics.decompose_sim3(T); Rb, tb, sb = metrics.decompose_sim3(R0)
print(sys.argv[2], "trial", sys.argv[1], "bit_equal", bool(np.array_equal(T, R0)), "max|dT| %.3g" % d,
      "rot %.3g deg" % metrics.rotation_error_deg(Ra, Rb), "trans %.3g m" % np.linalg.norm(ta - tb), "scale %.3g" % abs(sa / sb - 1), flush=True)
