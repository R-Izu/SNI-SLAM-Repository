"""R15 §3-1：縮退解を $\rho_Q$ の事前登録テストにかける。

**予測 D1**：縮退解の $\rho_Q$ は、正しい候補の $\rho_Q$ より明確に低い（1桁程度）
**予測 D2**：縮退解の $\rho_P$ は高い（$\rho_P$ が縮小を報酬することの直接確認）
**予測 P3**（impl）：縮退解の $\rho_Q$ は絶対値で 0.05 未満

**対照の取り方**：`office_0` の `ne` 隅では手法の解が潰れる。**同じ切り出し参照に対して**、
潰れた解と「正しい姿勢」の両方を採点すれば、参照・分母・ゲートが完全に揃う。

**「正しい姿勢」に何を使うか**：`T_gt.json` は**リポジトリに存在しない**
（Section C の合成 GT は手法自身の無摂動出力であり、ファイルとして書かれていない）。
そこで **被覆 100%（切り出しなし）で得た健全解 $T_{100}$** を対照に使う。
**参照の切り出しは座標系を変えない**ので、$T_{100}$ をそのまま `ne` 参照で採点できる。

> **$T_{100}$ は手法自身の出力であって独立な GT ではない。**
> **したがって本測定は「潰れた解 vs 手法が健全に出した解」の比較であり、
> 「潰れた解 vs 真値」ではない。** D1/D2 の判定はこの限定のもとで行う。
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

from bidirectional_score import score_pair                    # noqa: E402
from failure_decomposition import provenance                  # noqa: E402
from regbim import io_utils, metrics                          # noqa: E402
from regbim.methods import get_method                         # noqa: E402

CELLS = [(0.30, [1.0, 1.0], "ne"), (0.50, [1.0, 1.0], "ne"),
         (0.30, [-1.0, -1.0], "sw"), (0.50, [-1.0, -1.0], "sw")]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default="Registration/configs/sectionC/office_0.yaml")
    ap.add_argument("--out", default="Registration/output/diag/degenerate_rho.json")
    args = ap.parse_args()

    # 対照：被覆 100%（切り出しなし）の健全解。**手法自身の出力である。**
    base = yaml.safe_load(open(args.scene))
    base.setdefault("proposed", {})["scale_init"] = "median_axes"
    src0 = io_utils.load_source_cloud(base)
    T100 = get_method("proposed").register(src0, io_utils.load_reference_cloud(base), base)
    s100 = metrics.decompose_sim3(T100)[2]
    print("対照 T_100（被覆 100%%、切り出しなし）: s=%.4f  縮退=%s\n"
          % (s100, metrics.is_degenerate_sim3(T100)))
    if metrics.is_degenerate_sim3(T100):
        print("対照そのものが縮退している。この対照は使えない。")
        return 1

    out = []
    for keep, anchor, name in CELLS:
        cfg = yaml.safe_load(open(args.scene))
        cfg["reference"]["clip"] = {"keep_frac": keep, "anchor": anchor}
        cfg.setdefault("proposed", {})["scale_init"] = "median_axes"
        thresh = float(cfg["semantic_icp"]["max_corr_dist"])
        src = io_utils.load_source_cloud(cfg)
        dst = io_utils.load_reference_cloud(cfg)
        T = get_method("proposed").register(src, dst, cfg)
        s = metrics.decompose_sim3(T)[2]
        deg = metrics.is_degenerate_sim3(T)
        # **同じ参照・同じ分母・同じゲートで、この cell の解と T_100 を採点する**
        a = score_pair(src.points, src.labels, dst.points, dst.labels, T, thresh)
        b = score_pair(src.points, src.labels, dst.points, dst.labels, T100, thresh)
        rec = {"cell": "cov%03d_%s" % (int(keep * 100), name), "s": s,
               "degenerate": bool(deg),
               "rho_P_method": a["rho_P"], "rho_Q_method": a["rho_Q"],
               "rho_P_ref": b["rho_P"], "rho_Q_ref": b["rho_Q"],
               "n_reference": int(len(dst))}
        out.append(rec)
        print("=== %s (参照 %d 点) ===" % (rec["cell"], len(dst)))
        print("  この cell の解 : s=%.4e %s  rho_P=%.4f  rho_Q=%.4f"
              % (s, "**縮退**" if deg else "正常  ", a["rho_P"], a["rho_Q"]))
        print("  対照 T_100     :                   rho_P=%.4f  rho_Q=%.4f"
              % (b["rho_P"], b["rho_Q"]))
        if b["rho_Q"] > 0:
            print("  **rho_Q の比（この cell / 対照）= %.4f**" % (a["rho_Q"] / b["rho_Q"]))
        if b["rho_P"] > 0:
            print("  rho_P の比 = %.4f" % (a["rho_P"] / b["rho_P"]))
        print(flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"provenance": provenance(), "cells": out}, f, indent=2)
    print("wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
