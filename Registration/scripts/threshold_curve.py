"""R30 §2-2 — 成功率と閾値の関係を、**保存済みの変換行列から再採点して**出す。

**実行のやり直しをしない。** 既に保存した $T_{\rm est}$ を採点し直すだけである。

出すもの（R30 §2-2）
--------------------
| # | 内容 |
|---|---|
| 2-2-1 | 成功率–閾値曲線。**実データ 40 条件と GT-A の両方。** 0.02〜1.0 m |
| 2-2-2 | **$d_\Omega$ の分布そのもの**（5分位＋ヒストグラム）。曲線だけにしない |
| 2-2-3 | **6基準の3区分**（R18 §2 の形） |
| 2-2-4 | **回転・縮尺についても同じ曲線**。閾値は3つあるので並進だけ議論しない |

閾値の振り方
------------
**成功は3条件の AND である。** そこで **1つだけ振り、他の2つは事前登録値に固定する。**
（3つ同時に振ると、どの条件が効いているか分からなくなる。）

    並進を振る → 回転 5°・縮尺 0.05 に固定
    回転を振る → 並進 0.1 m・縮尺 0.05 に固定
    縮尺を振る → 並進 0.1 m・回転 5° に固定

**やらないこと（R30 §2-3）**
---------------------------
- **曲線を見てから閾値を決めない。** 本スクリプトは**記述しかしない**。閾値は main が別手順で決める
- **「この閾値なら何件通る」を根拠に閾値を選ばない**
- **3区分の「基準依存」を成功に数えない。** 別区分のまま残す

    conda activate sni-slam
    python Registration/scripts/threshold_curve.py
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
sys.path.insert(0, os.path.join(REPO, "Registration", "scripts"))
os.chdir(REPO)

from criterion_verdict import build_omega, d_omega          # noqa: E402
from failure_decomposition import provenance                # noqa: E402
from regbim import metrics                                  # noqa: E402

# 事前登録済みの閾値（R18 以来変えていない）。**振らない側はここに固定する。**
REG = {"rot_deg": 5.0, "trans": 0.1, "scale_ratio": 0.05}

# 走査範囲。**結果を見てから変えない。**
TRANS_GRID = np.concatenate([np.arange(0.02, 0.20, 0.01),
                             np.arange(0.20, 1.01, 0.05)])
ROT_GRID = np.array([0.5, 1, 2, 3, 5, 7.5, 10, 15, 20, 30, 45, 60, 90, 180])
SCALE_GRID = np.array([0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1,
                       0.15, 0.2, 0.3, 0.5, 1.0])

GT_A_SCENES = ["room_0", "room_1", "room_2", "office_0", "office_1",
               "office_2", "office_3", "office_4"]


def curve(rows: List[Dict], axis: str, grid) -> List[Dict]:
    """1軸だけ振り、他は事前登録値に固定して成功数を数える。

    `rows` の各要素は {"rot_deg", "d_omega", "scale_ratio", "degenerate"}。
    **`degenerate` は無条件に失敗**（stats.check_success と同じ扱い）。
    """
    out = []
    for v in grid:
        thr = dict(REG)
        thr[axis] = float(v)
        k = 0
        for r in rows:
            if r.get("degenerate"):
                continue
            if (r["rot_deg"] < thr["rot_deg"]
                    and r["d_omega"] < thr["trans"]
                    and r["scale_ratio"] < thr["scale_ratio"]):
                k += 1
        out.append({"threshold": float(v), "k": k, "n": len(rows),
                    "rate": k / len(rows) if rows else 0.0})
    return out


def quantiles(vals: List[float]) -> Dict:
    a = np.asarray([v for v in vals if np.isfinite(v)], dtype=float)
    if not len(a):
        return {}
    return {"n_finite": int(len(a)), "n_total": len(vals),
            "min": float(a.min()),
            "p20": float(np.percentile(a, 20)), "p40": float(np.percentile(a, 40)),
            "median": float(np.median(a)),
            "p60": float(np.percentile(a, 60)), "p80": float(np.percentile(a, 80)),
            "max": float(a.max())}


def histogram(vals: List[float], edges) -> List[Dict]:
    a = np.asarray([v for v in vals if np.isfinite(v)], dtype=float)
    h, _ = np.histogram(a, bins=edges)
    return [{"lo": float(edges[i]), "hi": float(edges[i + 1]), "count": int(h[i])}
            for i in range(len(h))]


# --------------------------------------------------------------------------- #
# 実データ：criterion_verdict.json から（6基準ぶんの値が条件ごとに入っている）
# --------------------------------------------------------------------------- #
def real_rows(path: str, criterion: int) -> List[Dict]:
    cv = json.load(open(path))
    rows = []
    for r in cv["results"]:
        for pc in r["per_criterion"]:
            if pc["id"] != criterion:
                continue
            rows.append({"target": r["target"], "rot_deg": pc["rot_deg"],
                         "d_omega": pc["d_omega"], "scale_ratio": pc["scale_ratio"],
                         "degenerate": False})
    return rows


def three_way(path: str) -> Dict:
    """R18 §2 の3区分。**「基準依存」を成功に数えない。**"""
    cv = json.load(open(path))
    c = {"all_pass": 0, "criterion_dependent": 0, "all_fail": 0}
    per = []
    for r in cv["results"]:
        oks = [bool(pc["ok"]) for pc in r["per_criterion"]]
        if all(oks):
            v = "all_pass"
        elif any(oks):
            v = "criterion_dependent"
        else:
            v = "all_fail"
        c[v] += 1
        per.append({"target": r["target"], "verdict": v,
                    "n_ok": sum(oks), "n_criteria": len(oks),
                    "d_omega_min": r.get("d_omega_min"),
                    "d_omega_max": r.get("d_omega_max")})
    return {"counts": c, "per_condition": per}


# --------------------------------------------------------------------------- #
# GT-A：保存済みの T_est と P から d_Ω を計算し直す
# --------------------------------------------------------------------------- #
def gt_a_rows(method: str, scenes: List[str]) -> List[Dict]:
    rows = []
    for s in scenes:
        p = "output/Registration/gt_a/%s/trial_matrices.json" % s
        if not os.path.exists(p):
            print("  %s: trial_matrices が無い。飛ばす" % s)
            continue
        tm = json.load(open(p))
        G = np.asarray(json.load(open(tm["T_gt"]["path"]))["T_gt"],
                       dtype=np.float64).reshape(4, 4)
        cfg = yaml.safe_load(open("Registration/configs/gt_a/%s.yaml" % s))
        omega = build_omega(cfg, seed=0)
        n = 0
        for t in tm["trials"]:
            if t["method"] != method or t["trial"] < 0:   # -1 は無摂動の T0
                continue
            T = np.asarray(t["T_est"], dtype=np.float64)
            P = np.asarray(t["P"], dtype=np.float64)
            expected = G @ metrics.invert_sim3(P)
            e = metrics.sim3_errors(T, expected)
            rows.append({"scene": s, "trial": t["trial"],
                         "rot_deg": e["rot_deg"],
                         "d_omega": (float("inf") if e.get("non_finite")
                                     else d_omega(T, expected, omega)),
                         "scale_ratio": e["scale_ratio"],
                         "degenerate": bool(e["degenerate"])})
            n += 1
        print("  %-10s %d 試行" % (s, n))
    return rows


def show(name: str, rows: List[Dict], edges) -> Dict:
    print("\n### %s（n=%d）" % (name, len(rows)))
    q = quantiles([r["d_omega"] for r in rows])
    if q:
        print("  d_Ω 分布: 最小 %.4f / 20%% %.4f / 中央 %.4f / 80%% %.4f / 最大 %.4f m"
              % (q["min"], q["p20"], q["median"], q["p80"], q["max"]))
        if q["n_finite"] < q["n_total"]:
            print("  （非有限 %d 件を分位から除外。曲線では失敗として数えている）"
                  % (q["n_total"] - q["n_finite"]))
    hist = histogram([r["d_omega"] for r in rows], edges)
    print("  ヒストグラム:")
    for b in hist:
        if b["count"]:
            print("    %6.2f – %6.2f m : %s %d"
                  % (b["lo"], b["hi"], "#" * min(40, b["count"]), b["count"]))
    cs = {a: curve(rows, a, g) for a, g in (("trans", TRANS_GRID),
                                            ("rot_deg", ROT_GRID),
                                            ("scale_ratio", SCALE_GRID))}
    print("  並進閾値 [m] → 成功率（回転 5°・縮尺 0.05 に固定）:")
    for c in cs["trans"]:
        if c["threshold"] in (0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0) or \
           abs(c["threshold"] - round(c["threshold"], 2)) < 1e-9 and c["threshold"] in (0.1,):
            print("    %.2f : %3d/%3d = %.1f%%"
                  % (c["threshold"], c["k"], c["n"], 100 * c["rate"]))
    print("  回転閾値 [度] → 成功率（並進 0.1 m・縮尺 0.05 に固定）:")
    for c in cs["rot_deg"]:
        if c["threshold"] in (1, 5, 15, 45, 180):
            print("    %5.1f : %3d/%3d = %.1f%%"
                  % (c["threshold"], c["k"], c["n"], 100 * c["rate"]))
    print("  縮尺閾値 → 成功率（並進 0.1 m・回転 5° に固定）:")
    for c in cs["scale_ratio"]:
        if c["threshold"] in (0.01, 0.05, 0.1, 0.3, 1.0):
            print("    %.3f : %3d/%3d = %.1f%%"
                  % (c["threshold"], c["k"], c["n"], 100 * c["rate"]))
    return {"n": len(rows), "quantiles": q, "histogram": hist, "curves": cs}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verdict", default="Registration/output/diag/criterion_verdict.json")
    ap.add_argument("--criterion", type=int, default=1,
                    help="実データの曲線に使う基準。既定 1（R18 の主基準）")
    ap.add_argument("--methods", nargs="*",
                    default=["proposed", "proposed_no_gravity", "proposed_fixed_scale",
                             "baseline_open3d"])
    ap.add_argument("--out", default="Registration/output/diag/threshold_curve.json")
    args = ap.parse_args()

    out: Dict = {"provenance": provenance(), "registered_thresholds": REG,
                 "note": "記述のみ。**曲線から閾値を決めていない**（R30 §2-3）",
                 "sections": {}}

    print("=" * 78)
    print("## 実データ 40 条件（基準 %d で採点）" % args.criterion)
    print("=" * 78)
    rr = real_rows(args.verdict, args.criterion)
    out["sections"]["realdata"] = show("実データ", rr,
                                       np.array([0, .1, .25, .5, 1, 2, 5, 10, 15, 20, 30, 50]))

    print("\n" + "=" * 78)
    print("## 6基準の3区分（R18 §2 の形）—— **「基準依存」は成功に数えない**")
    print("=" * 78)
    tw = three_way(args.verdict)
    out["sections"]["three_way"] = tw
    for k, v in tw["counts"].items():
        print("  %-22s %d / %d 条件" % (k, v, len(tw["per_condition"])))

    print("\n" + "=" * 78)
    print("## GT-A（合成）")
    print("=" * 78)
    for m in args.methods:
        print("\n%s の試行を読む:" % m)
        gr = gt_a_rows(m, GT_A_SCENES)
        if gr:
            out["sections"]["gt_a_%s" % m] = show("GT-A / %s" % m, gr,
                                                  np.array([0, .05, .1, .25, .5, 1, 2, 5, 10, 20]))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("\n\nwrote %s" % args.out)
    print("**曲線は記述である。閾値の決定は main が別手順で行う（R30 §2-4）。**")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
