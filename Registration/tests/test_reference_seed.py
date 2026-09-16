"""参照点群の標本化が、シードを渡したときだけ決定的になること（R30 §4）。

**なぜ要るか**：`_load_replica_reference` はシード無しで表面標本化しており、
**読むたびに別の 20 万点が出ていた**（同じ添字の点が中央 4.27 m 離れる）。
**Replica のどの実行も、一度もビット再現できていなかった。**

**測ってから固定した**（R30 §4）。参照の標本だけを変えた 10 実行で、
成功率は 1.000 から動かず、$d_\Omega$ の中央値は 0.00109 m の幅に収まった。

ここで守るのは2つ：
1. **シードを渡せば決定的である**
2. **シードを渡さなければ従来どおり**（既存 config の結果を静かに変えない）
"""

from __future__ import annotations

import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
os.chdir(REPO)

from regbim import io_utils  # noqa: E402

CFG = "Registration/configs/sectionC/room_0.yaml"
FAILS = []


def check(name, ok, extra=""):
    print("  %-4s %s%s" % ("OK" if ok else "NG", name, ("  " + extra) if extra else ""))
    if not ok:
        FAILS.append(name)


def load(seed=None):
    cfg = yaml.safe_load(open(CFG))
    if seed is not None:
        cfg["reference"] = dict(cfg["reference"], seed=seed)
    return io_utils.load_reference_cloud(cfg)


def main() -> int:
    if not os.path.exists(CFG):
        print("SKIP: %s が無い" % CFG)
        return 0

    print("1. シードを渡すと決定的")
    a, b = load(seed=0), load(seed=0)
    check("同じシードで点が完全一致",
          a.points.shape == b.points.shape and np.array_equal(a.points, b.points))
    check("同じシードでラベルも一致", np.array_equal(a.labels, b.labels))
    check("meta に sample_seed が残る", a.meta.get("sample_seed") == 0,
          "sample_seed=%s" % a.meta.get("sample_seed"))

    print("2. 違うシードなら違う標本（固定が効いていることの裏づけ）")
    c = load(seed=1)
    check("seed 0 と seed 1 は一致しない", not np.array_equal(a.points, c.points))

    print("3. シードを渡さなければ従来どおり（非決定的）")
    d, e = load(), load()
    check("既定では一致しない（既存 config の挙動を変えていない）",
          not np.array_equal(d.points, e.points))
    check("既定の meta は sample_seed=-1", d.meta.get("sample_seed") == -1,
          "sample_seed=%s" % d.meta.get("sample_seed"))

    print("4. 大域の乱数状態を汚さない")
    np.random.seed(12345)
    before = np.random.rand(3)
    np.random.seed(12345)
    load(seed=7)
    after = np.random.rand(3)
    check("参照の読み込みが numpy の大域状態を進めない",
          np.array_equal(before, after),
          "before=%s after=%s" % (np.round(before, 6), np.round(after, 6)))

    print()
    if FAILS:
        print("FAILED: %s" % ", ".join(FAILS))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
