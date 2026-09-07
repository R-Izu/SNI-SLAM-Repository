"""R15 §4：縮退解の走査を T3 の外へ広げる。

**指紋（全試行で `scale_ratio` が厳密に 0）だけに頼らない。**
それは縮退の帰結のひとつに過ぎず、参照側が縮退していない場合には現れない。
**保存された Sim(3) があれば `s < 1e-4` を直接見る。**

走査対象：
  - `trials.csv` を持つ全ディレクトリ（Section C・Phase 1〜5・T3・実データ）
  - `trial_matrices.json`（行列がある場合は s を直接読む）
  - 160 実行の sweep（`yaw_candidate_T` の全候補も見る）

**結果がゼロでも報告する**（走査した事実が要る）。
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from typing import Dict, List

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "Registration"))
os.chdir(REPO)

from regbim.metrics import SIM3_MIN_SCALE, decompose_sim3, is_degenerate_sim3  # noqa: E402


def scale_of(T) -> float:
    return decompose_sim3(np.asarray(T, dtype=np.float64))[2]


def scan_trials_csv(root: str) -> List[Dict]:
    """`scale_ratio` が全試行で厳密に 0 という指紋を探す（行列が無い出力向け）。"""
    out = []
    for f in sorted(glob.glob(os.path.join(root, "**", "trials.csv"), recursive=True)):
        rows = list(csv.DictReader(open(f)))
        if not rows:
            continue
        key = ("scale_ratio" if "scale_ratio" in rows[0]
               else "selfconsistency_scale_ratio"
               if "selfconsistency_scale_ratio" in rows[0] else None)
        if key is None:
            continue
        sr = np.array([float(r[key]) for r in rows])
        meth = np.array([r.get("method", "?") for r in rows])
        # **指紋（scale_ratio が全 0）は、縮尺を振っていない実験では偽陽性になる。**
        # 剛体手法（FGR / RANSAC）を摂動 scale = 1.0 で評価すれば、
        # scale_ratio は正しく恒等的に 0 になる。`journal_phase3` の 8 本がこれだった。
        # **縮尺を実際に振っている実験でのみ、指紋を「潰れ」として数える。**
        ps = np.array([float(r.get("pert_scale", 1.0)) for r in rows])
        scale_varies = bool(ps.max() - ps.min() > 1e-9)
        # `degenerate` 列があるなら、それが正である（新しい出力）
        deg_col = None
        if "degenerate" in rows[0]:
            deg_col = int(sum(r["degenerate"] == "True" for r in rows))
        # **参照側が健全なまま推定側だけ潰れた場合、指紋（全 0）は出ない。**
        # そのときは scale_ratio = |s/s_ref - 1| が 1 に近づく（s→0 だから）。
        # 行列が保存されていない古い出力でも、これなら直接 s の潰れを拾える。
        # **`scale_ratio` は |s/s_ref - 1| なので、縮小と拡大の両方が大きな値になる。**
        # 方向を分けないと、s/s_ref = 71 の**拡大**を「潰れ」と読んでしまう。
        #   縮小（潰れ）: s/s_ref -> 0  ==> sr -> 1（1 を超えない）
        #   拡大        : s/s_ref >> 1  ==> sr >> 1
        rec = {"path": os.path.relpath(f, REPO), "n": len(rows),
               "scale_varies": scale_varies,
               "all_scale_ratio_zero": bool((sr == 0.0).all() and scale_varies),
               "all_zero_but_rigid": bool((sr == 0.0).all() and not scale_varies),
               "n_scale_ratio_zero": int((sr == 0.0).sum()),
               "n_collapse": int(((sr > 0.99) & (sr <= 1.0)).sum()),
               "n_blowup": int((sr > 1.0).sum()),
               "max_scale_ratio": float(sr.max()),
               "degenerate_column": deg_col,
               # R15 §4：**baseline にも縮退がありうる。** 手法別に分けないと
               # 「比較の前提が変わっているか」が言えない。
               "by_method": {m: {
                   "n": int((meth == m).sum()),
                   "collapse": int(((meth == m) & (sr > 0.99) & (sr <= 1.0)).sum()),
                   "blowup": int(((meth == m) & (sr > 1.0)).sum()),
                   "zero": int(((meth == m) & (sr == 0.0)).sum()),
               } for m in sorted(set(meth.tolist()))}}
        # GT 基準の列があるならそちらでも見る（参照が手法自身の出力でないため確実）
        if "gt_scale_ratio" in rows[0]:
            g = np.array([float(r["gt_scale_ratio"]) for r in rows])
            rec["n_gt_collapse"] = int(((g > 0.99) & (g <= 1.0)).sum())
            rec["n_gt_blowup"] = int((g > 1.0).sum())
            rec["max_gt_scale_ratio"] = float(g.max())
        out.append(rec)
    return out


def scan_matrices(root: str) -> List[Dict]:
    """保存済み Sim(3) の s を直接見る（指紋に頼らない本命の走査）。"""
    out = []
    for f in sorted(glob.glob(os.path.join(root, "**", "trial_matrices.json"),
                              recursive=True)):
        try:
            blob = json.load(open(f))
        except Exception:
            continue
        recs = blob.get("trials", blob) if isinstance(blob, dict) else blob
        if not isinstance(recs, list):
            continue
        s = [scale_of(r["T_est"]) for r in recs if isinstance(r, dict) and "T_est" in r]
        if not s:
            continue
        a = np.array(s)
        out.append({"path": os.path.relpath(f, REPO), "n": len(a),
                    "n_degenerate": int((a < SIM3_MIN_SCALE).sum()),
                    "s_min": float(a.min()), "s_med": float(np.median(a)),
                    "s_max": float(a.max())})
    return out


def scan_sweeps(paths: List[str]) -> List[Dict]:
    """sweep の最終解と、**全候補**の s を見る。"""
    out = []
    for p in paths:
        try:
            blob = json.load(open(p))
        except Exception:
            continue
        rows = blob.get("rows") or []
        fin, cand = [], []
        for r in rows:
            if "T" in r:
                fin.append(scale_of(r["T"]))
            for T in (r.get("yaw_candidate_T") or []):
                cand.append(scale_of(T))
        if not fin and not cand:
            continue
        rec = {"path": os.path.relpath(p, REPO), "n_final": len(fin),
               "n_candidates": len(cand)}
        for tag, a in (("final", np.array(fin)), ("cand", np.array(cand))):
            if a.size:
                rec["%s_n_degenerate" % tag] = int((a < SIM3_MIN_SCALE).sum())
                rec["%s_s_min" % tag] = float(a.min())
                rec["%s_s_med" % tag] = float(np.median(a))
        out.append(rec)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="Registration/output")
    ap.add_argument("--out", default="Registration/output/diag/degenerate_scan.json")
    args = ap.parse_args()

    print("縮退の判定: s < %.0e（`is_degenerate_sim3` と同じ）" % SIM3_MIN_SCALE)
    print()

    csvs = scan_trials_csv(args.root)
    mats = scan_matrices(args.root)
    sweeps = scan_sweeps(sorted(glob.glob(
        os.path.join(args.root, "diag", "sweep_*.json"))))

    print("=== 1. trials.csv の指紋走査（scale_ratio が全試行で厳密に 0）===")
    print("  走査した trials.csv: %d 本 / 試行 %d 件"
          % (len(csvs), sum(c["n"] for c in csvs)))
    hits = [c for c in csvs if c["all_scale_ratio_zero"]]
    print("  **該当: %d 本**" % len(hits))
    for c in hits:
        print("    %-58s n=%d" % (c["path"], c["n"]))
    part = [c for c in csvs if not c["all_scale_ratio_zero"] and c["n_scale_ratio_zero"]]
    print("  （一部だけ 0 の出力: %d 本。丸めで 0 になったものを含むので指紋にはしない）"
          % len(part))
    rigid = [c for c in csvs if c["all_zero_but_rigid"]]
    if rigid:
        print("  **偽陽性として除外: %d 本 / 試行 %d 件**"
              % (len(rigid), sum(c["n"] for c in rigid)))
        print("    縮尺を振っていない実験（摂動 scale = 1.0 固定）では、"
              "剛体手法の scale_ratio は正しく恒等的に 0 になる。")
        for c in rigid[:3]:
            print("      %s" % c["path"])
        if len(rigid) > 3:
            print("      …ほか %d 本" % (len(rigid) - 3))
    print()

    print("=== 1-b. 推定側だけが潰れた場合の走査 ===")
    print("  指紋（全 0）は**参照側も潰れたとき**にしか出ない。")
    print("  s→0 なら |s/s_ref - 1| → 1 に寄るので、0.99 < sr <= 1.0 が縮小の指標。")
    print("  **sr > 1.0 は拡大であって縮小ではない。別に数える。**")
    ntot = sum(c["n"] for c in csvs)
    print("  **縮小（0.99 < sr <= 1.0）: %d 件 / %d 件**"
          % (sum(c["n_collapse"] for c in csvs), ntot))
    for c in sorted(csvs, key=lambda x: -x["n_collapse"])[:5]:
        if c["n_collapse"]:
            print("    %-58s %d/%d" % (c["path"], c["n_collapse"], c["n"]))
    print("  **拡大（sr > 1.0）: %d 件 / %d 件**"
          % (sum(c["n_blowup"] for c in csvs), ntot))
    for c in sorted(csvs, key=lambda x: -x["n_blowup"])[:5]:
        if c["n_blowup"]:
            print("    %-58s %d/%d (最大 s/s_ref = %.1f 倍)"
                  % (c["path"], c["n_blowup"], c["n"], c["max_scale_ratio"] + 1.0))
    gt = [c for c in csvs if "n_gt_collapse" in c]
    if gt:
        print("  GT 基準の列を持つ出力 %d 本: 縮小 %d 件 / 拡大 %d 件"
              % (len(gt), sum(c["n_gt_collapse"] for c in gt),
                 sum(c["n_gt_blowup"] for c in gt)))
    else:
        print("  GT 基準の列を持つ出力は無い")
    print()

    print("=== 1-c. 手法別（R15 §4：baseline にも縮退がありうる）===")
    agg: Dict[str, Dict[str, int]] = {}
    for c in csvs:
        for m, v in c.get("by_method", {}).items():
            a = agg.setdefault(m, {"n": 0, "collapse": 0, "blowup": 0, "zero": 0})
            for k in a:
                a[k] += v[k]
    for m, v in sorted(agg.items(), key=lambda kv: -kv[1]["n"]):
        print("  %-16s n=%-6d 縮小 %-4d 拡大 %-4d 指紋(sr==0) %d"
              % (m, v["n"], v["collapse"], v["blowup"], v["zero"]))
    print()

    print("=== 1-d. 出力の系統別 ===")
    groups = {"実データ 40 条件": "output/realdata/",
              "T3 v2 (208 cell)": "output/t3_coverage_v2/",
              "T3 v1": "output/t3_coverage/",
              "Section C ほか": None}
    for label, pref in groups.items():
        sel = ([c for c in csvs if pref in c["path"].replace("\\", "/")] if pref else
               [c for c in csvs if not any(
                   p and p in c["path"].replace("\\", "/") for p in groups.values())])
        if not sel:
            print("  %-20s （該当なし）" % label)
            continue
        print("  %-20s ファイル %-4d 試行 %-6d 縮小 %-4d 拡大 %-4d 指紋 %d"
              % (label, len(sel), sum(c["n"] for c in sel),
                 sum(c["n_collapse"] for c in sel), sum(c["n_blowup"] for c in sel),
                 sum(1 for c in sel if c["all_scale_ratio_zero"])))
    print()

    print("=== 2. 保存済み行列の s を直接見る ===")
    if not mats:
        print("  trial_matrices.json が見つからない（R7 §3-2 以降の出力にしか無い）")
    for m in mats:
        flag = "**縮退 %d**" % m["n_degenerate"] if m["n_degenerate"] else "縮退 0"
        print("  %-52s n=%-5d %s  s: min %.4g / 中央 %.4g / max %.4g"
              % (m["path"], m["n"], flag, m["s_min"], m["s_med"], m["s_max"]))
    print("  合計: 行列 %d 件中 縮退 %d 件"
          % (sum(m["n"] for m in mats), sum(m["n_degenerate"] for m in mats)))
    print()

    print("=== 3. 160 実行の sweep（最終解と全候補）===")
    for s in sweeps:
        print("  %-46s 最終 %d 件(縮退 %d, s中央 %.4g) / 候補 %d 件(縮退 %d, s最小 %.4g)"
              % (os.path.basename(s["path"]), s["n_final"],
                 s.get("final_n_degenerate", 0), s.get("final_s_med", float("nan")),
                 s["n_candidates"], s.get("cand_n_degenerate", 0),
                 s.get("cand_s_min", float("nan"))))
    tot_c = sum(s["n_candidates"] for s in sweeps)
    tot_cd = sum(s.get("cand_n_degenerate", 0) for s in sweeps)
    print("  合計: 候補 %d 件中 縮退 %d 件" % (tot_c, tot_cd))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"min_scale": SIM3_MIN_SCALE, "trials_csv": csvs,
                   "matrices": mats, "sweeps": sweeps}, f, indent=2)
    print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
