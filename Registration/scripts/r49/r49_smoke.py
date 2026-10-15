"""R49 段階1 の前の動作確認：GT-A room_0 の trial 0 を、指定した方式で 1 回（成否は見ない。動くか・時間・候補の数だけ）"""
import json, os, sys, time
import numpy as np
import yaml

REPO = os.path.expanduser("~/rizu/SNI-SLAM")
sys.path.insert(0, os.path.join(REPO, "Registration")); os.chdir(REPO)
from regbim import io_utils, metrics, instrument                 # noqa: E402
from regbim.methods import get_method                            # noqa: E402

meth = sys.argv[1]; extra = sys.argv[2] if len(sys.argv) > 2 else ""
cfg = yaml.safe_load(open("Registration/configs/gt_a_plan_a/room_0.yaml"))
if extra == "lf_off":
    cfg["proposed"]["label_filter"] = {"enabled": False}
tm = json.load(open("output/Registration/r45/plan_a/room_0/trial_matrices.json"))
row = [t for t in tm["trials"] if t["trial"] == 0][0]
src = io_utils.load_source_cloud(cfg); dst = io_utils.load_reference_cloud(cfg)
instrument.start(); t0 = time.perf_counter()
T = get_method(meth).register(metrics.transform_cloud(src, np.asarray(row["P"])), dst, cfg)
dt = time.perf_counter() - t0; rec = instrument.stop()
tags = {}
for c in rec["icp"]:
    tags[c["tag"]] = tags.get(c["tag"], 0) + 1
print(meth, extra, "time %.1f s" % dt, "icp calls", tags, "stages", {k: round(v, 1) for k, v in rec["stages"].items()},
      "winner", rec.get("winner_index"), "finite", bool(np.isfinite(T).all()))
