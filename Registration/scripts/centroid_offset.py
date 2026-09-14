"""R25 第2稿 §4-3 の**事後の探索**：並進の初期化が何 m ずれているかを直に測る。

**これは予測ではない。** V2 が外れた後に「では並進はどれだけずれているか」を
見るために回した。**事前登録していないので、当たり外れを主張しない**
（`_IMPL_OPERATING_RULES.md` F2）。

種の並進は「source の構造重心を reference の構造重心へ持っていく」。
廊下スキャンは室 411 を含みつつ廊下へ長く伸びているので、
**その重心は室の重心から、廊下の伸びた分だけ離れている**はずである。
その距離を、GT で正しく置いたときの構造重心のずれとして出す。

    python Registration/scripts/centroid_offset.py --targets m3_cor_a__E2 ...
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

from failure_decomposition import provenance          # noqa: E402
from regbim import io_utils, metrics                  # noqa: E402
from regbim.labels import NAME_TO_ID                  # noqa: E402

STRUCT = ("wall", "floor", "ceiling")
_LO, _HI = 2.0, 98.0


def struct(pts, labels):
    ids = [NAME_TO_ID[n] for n in STRUCT if n in NAME_TO_ID]
    return pts[np.isin(labels, ids)]


def span(p):
    lo, hi = np.percentile(p, [_LO, _HI], axis=0)
    return hi - lo


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--targets", nargs="+", required=True)
    ap.add_argument("--gt-kit", default="output/GT_alignment_probe")
    ap.add_argument("--out", default="Registration/output/diag/centroid_offset.json")
    args = ap.parse_args()

    rows = []
    print("%-16s %28s %22s %10s" % ("target", "source 構造の広がり[m]",
                                    "BIM の広がり[m]", "重心差[m]"))
    for t in args.targets:
        cfg = yaml.safe_load(open("Registration/configs/realdata/%s.yaml" % t))
        cfg["source"] = dict(cfg["source"], seed=0)
        src = io_utils.load_source_cloud(cfg)
        dst = io_utils.load_reference_cloud(cfg)
        G = np.asarray(json.load(open(os.path.join(
            args.gt_kit, "T_gt", "T_gt_%s.json" % t.split("__")[0])))["T_gt"],
            dtype=np.float64).reshape(4, 4)
        sp = metrics.apply_sim3(G, struct(src.points, src.labels))   # BIM 座標
        dp = struct(dst.points, dst.labels)
        d = float(np.linalg.norm(sp.mean(axis=0) - dp.mean(axis=0)))
        ss, ds = span(sp), span(dp)
        rows.append({"target": t, "src_span_m": ss.tolist(), "dst_span_m": ds.tolist(),
                     "centroid_offset_m": d,
                     "src_long_axis_m": float(ss[:2].max()),
                     "dst_long_axis_m": float(ds[:2].max())})
        print("%-16s  %6.2f %6.2f %6.2f      %6.2f %6.2f %6.2f    %8.2f"
              % (t, ss[0], ss[1], ss[2], ds[0], ds[1], ds[2], d))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"provenance": provenance(), "rows": rows}, f, indent=2,
                  ensure_ascii=False)
    print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
