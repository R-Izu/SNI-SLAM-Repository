"""R11 §3 / R12 §4 — 失敗を3つに分けて数える。

**「探索の問題か、目的関数の問題か」という二択が、3回の混乱を生んだ。**
分けて数える。各実行を次のどれかに分類する：

| # | 分類 | 判定 | 直す場所 |
|---|---|---|---|
| (i) | **候補が生成されていない** | 候補の回転のうち GT に 5° 以内のものが**無い** | 候補生成 |
| (ii) | **候補はあるが収束していない** | GT に近い候補はあるが、**その ICP 後の解が GT から遠い** | 収束 |
| (iii) | **収束したが順位で負けた** | GT 近傍に収束した候補があるのに、**別候補が高得点** | 目的関数 |
| (ok) | 成功 | 勝った候補の解が GT 近傍 | — |

**閾値は明示して報告する。** (ii)/(iii) を分ける「GT 近傍」の定義は
成功閾値（並進 0.1 m）ではなく、**吸引域に入ったか**を見たいので広く取る。
既定 1.0 m は、実際の失敗（11〜20 m）より1桁小さく、
基準の広がり（§3-2 で最大 0.433 m）より大きい値として選んだ。
**この選び方も報告する。** 併せて距離の分布そのものを出すので、閾値を変えて読み直せる。

    conda activate sni-slam
    python Registration/scripts/failure_decomposition.py \
        --sweeps Registration/output/diag/sweep_inner.json Registration/output/diag/sweep_both.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np
import yaml

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "Registration"))
os.chdir(REPO)

from regbim.metrics import decompose_sim3, rotation_error_deg    # noqa: E402


def wilson(k: int, n: int, z: float = 1.96):
    if n <= 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def classify(row: Dict, G: np.ndarray, rot_tol: float, basin: float) -> Dict:
    cR = row.get("yaw_candidate_R")
    cT = row.get("yaw_candidate_T")
    if not cR or not cT:
        return {"class": "no_diag"}
    RG = decompose_sim3(G)[0]
    rot = [rotation_error_deg(np.asarray(R), RG) for R in cR]
    near = [i for i, r in enumerate(rot) if r < rot_tol]
    if not near:
        return {"class": "i_no_candidate", "min_cand_rot_deg": float(min(rot))}
    # GT に近い候補のうち、ICP 後に GT へいちばん寄ったもの
    d = {}
    for i in near:
        T = np.asarray(cT[i], dtype=np.float64)
        d[i] = float(np.linalg.norm((T[:3, 3] - G[:3, 3])))
    best = min(d, key=d.get)
    winner = row.get("yaw_winner")
    if d[best] > basin:
        return {"class": "ii_no_convergence", "best_near_idx": best,
                "best_near_trans_m": d[best], "min_cand_rot_deg": float(min(rot))}
    if winner == best:
        return {"class": "ok", "best_near_idx": best, "best_near_trans_m": d[best]}
    return {"class": "iii_lost_on_score", "best_near_idx": best,
            "best_near_trans_m": d[best], "winner_idx": winner,
            "scores": row.get("yaw_candidate_scores")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweeps", nargs="+", required=True)
    ap.add_argument("--rot-tol", type=float, default=5.0)
    ap.add_argument("--basin", type=float, default=1.0)
    ap.add_argument("--out", default="Registration/output/diag/failure_decomposition.json")
    args = ap.parse_args()

    LAB = {"i_no_candidate": "(i) 候補が無い",
           "ii_no_convergence": "(ii) 収束していない",
           "iii_lost_on_score": "(iii) 順位で負けた",
           "ok": "(ok) 成功",
           "no_diag": "診断データ無し"}

    print("判定の閾値: 候補の回転 < %.1f度 を『GT に近い候補』、"
          "ICP 後の並進 < %.2f m を『吸引域に入った』とする" % (args.rot_tol, args.basin))
    print()
    out: List[Dict] = []
    for path in args.sweeps:
        blob = json.load(open(path))
        cfg = yaml.safe_load(open(blob["config"]))
        G = np.asarray(json.load(open(cfg["eval"]["t_gt_path"]))["T_gt"],
                       dtype=np.float64)
        inner = blob.get("inner_only")
        for mode in sorted({r["mode"] for r in blob["rows"]}):
            rows = [r for r in blob["rows"] if r["mode"] == mode]
            cls = [classify(r, G, args.rot_tol, args.basin) for r in rows]
            n = len(cls)
            counts: Dict[str, int] = {}
            for c in cls:
                counts[c["class"]] = counts.get(c["class"], 0) + 1
            tag = "%s / %s" % ("室内面" if inner else "両面", mode)
            print("=== %s  (n=%d) ===" % (tag, n))
            for k in ("i_no_candidate", "ii_no_convergence", "iii_lost_on_score",
                      "ok", "no_diag"):
                if k not in counts:
                    continue
                lo, hi = wilson(counts[k], n)
                print("  %-22s %2d/%d = %.2f  [%.2f, %.2f]"
                      % (LAB[k], counts[k], n, counts[k] / n, lo, hi))
            mr = [c["min_cand_rot_deg"] for c in cls if "min_cand_rot_deg" in c]
            if mr:
                print("  候補の最小回転誤差: 中央値 %.2f度 / 範囲 [%.2f, %.2f]"
                      % (np.median(mr), min(mr), max(mr)))
            bn = [c["best_near_trans_m"] for c in cls if "best_near_trans_m" in c]
            if bn:
                print("  GT 近傍候補の ICP 後の並進: 中央値 %.3f m / 範囲 [%.3f, %.3f]"
                      % (np.median(bn), min(bn), max(bn)))
            print()
            out.append({"sweep": path, "inner_only": inner, "mode": mode,
                        "n": n, "counts": counts,
                        "rot_tol": args.rot_tol, "basin": args.basin,
                        "per_seed": [dict(c, seed=r["seed"])
                                     for c, r in zip(cls, rows)]})
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
