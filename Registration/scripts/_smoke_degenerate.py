"""潰れ拒否が (a) ne セルの偽の成功を消し (b) sw セルを一切変えないことを確認する。

`_evaluate` をそのまま呼ぶので、benchmark 本体と同じ経路を通る。
"""

from __future__ import annotations

import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.abspath("."), "Registration"))
sys.path.insert(0, os.path.join(os.path.abspath("."), "Registration", "scripts"))

from benchmark import _evaluate                      # noqa: E402
from regbim import io_utils                          # noqa: E402

TRIALS = 20

for anchor, name in ([1.0, 1.0], "ne"), ([-1.0, -1.0], "sw"):
    cfg = yaml.safe_load(open("Registration/configs/sectionC/office_0.yaml"))
    cfg["reference"]["clip"] = {"keep_frac": 0.30, "anchor": anchor}
    cfg.setdefault("proposed", {})["scale_init"] = "median_axes"
    cfg.setdefault("diagnostics", {})["record_yaw"] = True
    src = io_utils.load_source_cloud(cfg)
    dst = io_utils.load_reference_cloud(cfg)
    out = _evaluate("proposed", src, dst, None, cfg, TRIALS,
                    int(cfg["eval"]["seed"]))
    row, trials = out[0], out[1]
    n_deg = sum(bool(t["degenerate"]) for t in trials)
    n_ok = sum(bool(t["selfconsistency_success"]) for t in trials)
    rot = np.array([t["selfconsistency_rot_deg"] for t in trials])
    tr = np.array([t["selfconsistency_trans"] for t in trials])
    print("=== anchor=%s (n=%d) ===" % (name, TRIALS))
    print("  潰れた試行            : %d / %d" % (n_deg, TRIALS))
    print("  成功                  : %d / %d" % (n_ok, TRIALS))
    print("  row['n_degenerate']   : %s" % row["n_degenerate"])
    print("  回転誤差 中央値 %.3f度 / 並進誤差 中央値 %.5f m"
          % (np.median(rot), np.median(tr)))
    print()
