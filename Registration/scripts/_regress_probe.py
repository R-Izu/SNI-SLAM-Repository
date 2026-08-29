"""固定した入力で `proposed.register()` を1回呼び、返る Sim(3) を書き出す。

`regress_scale_init.sh` から、**変更前のコミット**と**現在の作業ツリー**の両方で呼ばれる。
入力を .npz で固定するのは、`io_utils._load_slam_mesh` の
`sample_points_uniformly()` が seed を取れず（Open3D 0.13）、
**実行ごとに違う点を引いてしまう**ため。入力を固定すれば `register()` は決定的である
（`rotation.py:58` が `default_rng(0)`）。

    --dump PATH   config から点群を作って PATH に保存する（1回だけ）
    --load PATH   PATH の点群を読んで register() を呼び、--out に結果を書く
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
os.chdir(REPO)

from regbim import io_utils                              # noqa: E402
from regbim.labels import LabeledCloud                   # noqa: E402
from regbim.methods import get_method                    # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--dump", default=None)
    ap.add_argument("--load", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))

    if args.dump:
        s = io_utils.load_source_cloud(cfg)
        d = io_utils.load_reference_cloud(cfg)
        os.makedirs(os.path.dirname(os.path.abspath(args.dump)), exist_ok=True)
        np.savez_compressed(
            args.dump,
            sp=s.points, sl=s.labels,
            sn=s.normals if s.normals is not None else np.zeros((0, 3)),
            dp=d.points, dl=d.labels,
            dn=d.normals if d.normals is not None else np.zeros((0, 3)))
        print("  source %d 点 / reference %d 点 -> %s" % (len(s), len(d), args.dump))
        return 0

    z = np.load(args.load, allow_pickle=False)
    src = LabeledCloud(points=z["sp"], labels=z["sl"],
                       normals=z["sn"] if len(z["sn"]) else None)
    dst = LabeledCloud(points=z["dp"], labels=z["dl"],
                       normals=z["dn"] if len(z["dn"]) else None)

    T = get_method("proposed").register(src, dst, cfg)
    out = {"T": np.asarray(T, dtype=np.float64).tolist()}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print("  Sim(3) -> %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
