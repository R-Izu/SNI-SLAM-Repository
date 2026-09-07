"""R13 §1 — 「ヨー誤りだが成功」46 件の元記録を出す。

候補は `Rr.T @ Rz(k*pi/2) @ Rs`（`rotation.py`）なので **厳密に 90° 間隔**である。
勝者が基準から 5° 以内なら、勝者は一意の最近候補になる。
**したがって success かつ picked_correct=False は定義上ありえない。**

**46 件が実在するなら、どこかで対応がずれている。** その場所を特定する。
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict, List

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "Registration"))
os.chdir(REPO)

from regbim.metrics import rotation_error_deg          # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="Registration/output/t3_coverage_v2")
    ap.add_argument("--out", default="Registration/output/diag/yaw46_audit.json")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.root, "*", "yaw_diag.json")))
    print("yaw_diag.json: %d 件" % len(files))
    n_all = n_wrong = n_wrong_succ = 0
    bad: List[Dict] = []
    spacing: List[float] = []
    for f in files:
        cell = os.path.basename(os.path.dirname(f))
        try:
            recs = json.load(open(f))
        except Exception:
            continue
        for r in recs:
            n_all += 1
            # 当時の列名（R7 の改名より前）
            pc = r.get("picked_correct", r.get("selfconsistency_picked_correct"))
            su = r.get("success", r.get("selfconsistency_success"))
            if pc is False:
                n_wrong += 1
                if su:
                    n_wrong_succ += 1
                    bad.append(dict(r, cell=cell))
            # 候補間の角度を確かめる（本当に 90 度刻みか）
            cR = r.get("candidate_R")
            if cR and len(cR) == 4 and len(spacing) < 200:
                A = [np.asarray(x) for x in cR]
                for i in range(4):
                    for j in range(i + 1, 4):
                        spacing.append(rotation_error_deg(A[i], A[j]))

    print("全試行 %d / picked_correct=False %d / **そのうち success %d**"
          % (n_all, n_wrong, n_wrong_succ))
    if spacing:
        s = np.array(spacing)
        print("候補間の角度: 最小 %.3f度 / 最大 %.3f度 / 一意な値 %s"
              % (s.min(), s.max(), sorted(set(np.round(s, 1)))[:6]))

    if not bad:
        print("\n該当 0 件。46 件はこの出力からは再現しない。")
    else:
        print("\n--- 該当行（先頭 10 件）---")
        for r in bad[:10]:
            print("  cell=%s trial=%s winner=%s correct=%s "
                  "winner_rot_err=%s best_possible=%s scores=%s"
                  % (r.get("cell"), r.get("trial"), r.get("winner"),
                     r.get("correct", r.get("selfconsistency_correct_idx")),
                     r.get("winner_rot_err_deg",
                           r.get("selfconsistency_winner_rot_err_deg")),
                     r.get("best_possible_rot_err_deg"),
                     r.get("candidate_scores")))
        # 勝者の回転誤差の分布。5 度以内なら定義上おかしい
        we = [r.get("winner_rot_err_deg",
                    r.get("selfconsistency_winner_rot_err_deg")) for r in bad]
        we = [x for x in we if x is not None]
        if we:
            a = np.array(we)
            print("\n該当行の『勝者の回転誤差』: 中央値 %.3f度 / 範囲 [%.3f, %.3f]"
                  % (np.median(a), a.min(), a.max()))
            print("  5度以内の件数: %d / %d" % (int((a < 5).sum()), len(a)))
            print("  → 5度を超えているなら、**成功判定は回転以外の条件で決まっていた**")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"n_all": n_all, "n_wrong": n_wrong,
                   "n_wrong_success": n_wrong_succ, "rows": bad}, f, indent=2)
    print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
