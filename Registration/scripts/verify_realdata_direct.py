"""実データの無摂動解が、点サンプリングの引き直しで再現するかを確かめる。

`realdata_results.json` の **GT 基準の量は `direct_*` だけ**で、しかも
**各条件 1 実行**である（`med_*` と `success_rate` は自己一貫性であって GT 基準ではない。
direct が 180 度ずれている 6 件でも med は 0.07〜0.63 度、成功率は最大 1.00）。

さらに実データ config は `source.seed` を持たないので、
`sample_points_uniformly` は既定の `seed=-1`（非決定的）で走っていた。
**つまり保存済みの `direct_*` は 1 ドローである。**

**確かめたいのは値の細かい振れではなく、90 度／180 度の「回転モード」が
実行ごとに変わるかである。** モードが変われば、シーン別の良し悪しの
読み取りそのものが 1 ドローの産物になる。
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import yaml

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "Registration"))
sys.path.insert(0, os.path.join(REPO, "Registration", "scripts"))
os.chdir(REPO)

from failure_decomposition import provenance                  # noqa: E402
from regbim import io_utils, metrics                          # noqa: E402
from regbim.methods import get_method                         # noqa: E402

TARGETS = ["m3_room_a__E1", "m3_room_a__E3", "m3_block_b__E3", "m3_cor_a__E3"]


def mode_of(deg: float) -> str:
    if deg < 5:
        return "正"
    if 60 < deg < 120:
        return "90度"
    if deg > 150:
        return "180度"
    return "他(%.0f度)" % deg


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--targets", nargs="+", default=TARGETS)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", default="Registration/output/diag/realdata_direct_repeat.json")
    args = ap.parse_args()

    stored = {("%s__%s" % (r["scene"], r["condition"])): r
              for r in json.load(open("Registration/output/realdata/realdata_results.json"))}

    out = []
    for name in args.targets:
        cfg = yaml.safe_load(open("Registration/configs/realdata/%s.yaml" % name))
        G = np.asarray(json.load(open(cfg["eval"]["t_gt_path"]))["T_gt"], dtype=np.float64)
        dst = io_utils.load_reference_cloud(cfg)
        s0 = stored.get(name)
        print("=== %s ===" % name)
        if s0:
            print("  保存済み（1 ドロー）: %.2f度 %.3fm 縮尺比 %.4f  → %s"
                  % (s0["direct_rot_deg"], s0["direct_trans"], s0["direct_scale_ratio"],
                     mode_of(s0["direct_rot_deg"])))
        rows = []
        for sd in args.seeds:
            c = dict(cfg, source=dict(cfg["source"], seed=sd))
            src = io_utils.load_source_cloud(c)
            T = get_method("proposed").register(src, dst, c)
            e = metrics.sim3_errors(T, G)
            rows.append({"seed": sd, "rot_deg": e["rot_deg"], "trans": e["trans"],
                         "scale_ratio": e["scale_ratio"], "mode": mode_of(e["rot_deg"]),
                         "degenerate": bool(e["degenerate"])})
            print("  seed=%d  %7.2f度 %8.3fm 縮尺比 %.4f  → %s"
                  % (sd, e["rot_deg"], e["trans"], e["scale_ratio"], rows[-1]["mode"]),
                  flush=True)
        modes = sorted({r["mode"] for r in rows})
        print("  **回転モード: %s（%s）**"
              % ("/".join(modes), "安定" if len(modes) == 1 else "**実行ごとに変わる**"))
        tr = np.array([r["trans"] for r in rows])
        print("  並進誤差: 中央値 %.3f m / 範囲 [%.3f, %.3f]\n"
              % (np.median(tr), tr.min(), tr.max()))
        out.append({"target": name, "stored": s0 and
                    {k: s0[k] for k in ("direct_rot_deg", "direct_trans",
                                        "direct_scale_ratio")},
                    "repeats": rows, "modes": modes})

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"provenance": provenance(), "results": out}, f, indent=2,
                  ensure_ascii=False)
    print("wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
