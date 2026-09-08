"""事前登録した判定規則を、双方向採点の結果に**そのまま**当てる。

登録済み（`2026-09-09_r14_preregistration_rho_q.md` §3、R16 §2 で境界訂正）：

  主判定：(iii) に分類された 33 実行のうち、**rho_Q 単独**で候補を順位付けたとき
          正しい候補が 1 位になる数 k について
            k >= 23        → 目的関数への追加を設計する
            11 <= k <= 22  → 判定保留（Wilson 95%CI が 0.5 を含む）
            k <= 10        → 別の原因を探す

副次（**判定には使わない**）：rho_P*rho_Q / min(rho_P,rho_Q) での順位、
(ii) での挙動、rho_P・rho_Q 対 s/s_G の順位相関（Spearman）。

**規則は測定前に確定している。ここでは当てるだけで、閾値を動かさない。**
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from typing import Dict, List

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "Registration"))
sys.path.insert(0, os.path.join(REPO, "Registration", "scripts"))
os.chdir(REPO)

from failure_decomposition import wilson                     # noqa: E402

DESIGN_AT, ABANDON_AT = 23, 10          # 事前登録の境界（n=33）


def rank_wins(cands: List[Dict], key) -> bool:
    """key を最大化する候補が「正しい候補」か。"""
    vals = [key(c) for c in cands]
    return bool(cands[int(np.argmax(vals))]["is_correct"])


def spearman(x, y):
    """順位相関。scipy に依存せず、順位のピアソン相関で出す。"""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3:
        return float("nan"), 0
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    d = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return (float((rx * ry).sum() / d) if d else float("nan")), len(x)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", default="Registration/output/diag/bidirectional_scores.json")
    args = ap.parse_args()

    blob = json.load(open(args.scores))
    recs = blob["records"]
    runs: Dict[tuple, List[Dict]] = collections.defaultdict(list)
    for r in recs:
        runs[(r["sweep"], r["mode"], r["seed"])].append(r)
    print("候補 %d 件 / 実行 %d 件" % (len(recs), len(runs)))
    by_class = collections.Counter(v[0]["run_class"] for v in runs.values())
    print("実行の分類: %s" % dict(by_class))
    print()

    iii = {k: v for k, v in runs.items() if v[0]["run_class"] == "iii_lost_on_score"}
    n = len(iii)
    print("=== 主判定：(iii) の %d 実行で rho_Q 単独が正しい候補を1位にするか ===" % n)
    if n == 0:
        print("  (iii) が 0 件。判定できない。")
        return 0

    k_q = sum(rank_wins(v, lambda c: c["rho_Q"]) for v in iii.values())
    lo, hi = wilson(k_q, n)
    print("  **k = %d / %d = %.3f  Wilson 95%%CI [%.3f, %.3f]**"
          % (k_q, n, k_q / n, lo, hi))
    verdict = ("**設計する**" if k_q >= DESIGN_AT else
               "**別の原因を探す**" if k_q <= ABANDON_AT else "**判定保留**")
    print("  事前登録の境界: k>=%d 設計する / %d<=k<=%d 保留 / k<=%d 別の原因"
          % (DESIGN_AT, ABANDON_AT + 1, DESIGN_AT - 1, ABANDON_AT))
    print("  → 判定: %s" % verdict)
    print()

    print("=== 副次（判定には使わない）===")
    for label, key in (("現行 rho_P 単独", lambda c: c["rho_P"]),
                       ("rho_P * rho_Q", lambda c: c["rho_P"] * c["rho_Q"]),
                       ("min(rho_P, rho_Q)", lambda c: min(c["rho_P"], c["rho_Q"])),
                       ("手法の実スコア", lambda c: (c["method_score"] or 0.0))):
        k = sum(rank_wins(v, key) for v in iii.values())
        l2, h2 = wilson(k, n)
        print("  %-18s k=%2d/%d = %.3f [%.3f, %.3f]" % (label, k, n, k / n, l2, h2))
    print()

    ii = {k: v for k, v in runs.items() if v[0]["run_class"] == "ii_no_convergence"}
    if ii:
        k2 = sum(rank_wins(v, lambda c: c["rho_Q"]) for v in ii.values())
        n2 = len(ii)
        # (ii) では正しい候補が吸引域に入っていないので、is_correct は原則 False
        n_has = sum(any(c["is_correct"] for c in v) for v in ii.values())
        print("  (ii) の %d 実行: rho_Q が正しい候補を1位にした %d 件"
              "（そもそも正しい候補を含む実行は %d 件）" % (n2, k2, n_has))
    print()

    print("=== 縮尺との関係（R16 §3-1。判定には使わない）===")
    s = [r["s_over_sG"] for r in recs]
    for tag in ("rho_P", "rho_Q"):
        rho, nn = spearman(s, [r[tag] for r in recs])
        print("  Spearman(%s, s/s_G) = %+.3f  (n=%d)" % (tag, rho, nn))
    print("  → rho_P が負・rho_Q が正なら、互いの病理を打ち消す関係（予測 P4）")
    print()

    print("=== 正しい候補と誤答の分布（散布図の数値版）===")
    cor = [r for r in recs if r["is_correct"]]
    wrong = [r for r in recs if not r["is_correct"]]
    for tag, sel in (("正しい候補", cor), ("誤答", wrong)):
        if not sel:
            print("  %-10s (0 件)" % tag)
            continue
        p = np.array([r["rho_P"] for r in sel])
        q = np.array([r["rho_Q"] for r in sel])
        print("  %-10s n=%-4d rho_P 中央 %.4f [%.4f, %.4f] / rho_Q 中央 %.4f [%.4f, %.4f]"
              % (tag, len(sel), np.median(p), p.min(), p.max(),
                 np.median(q), q.min(), q.max()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
