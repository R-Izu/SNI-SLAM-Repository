"""R22 §5-1 — K1 を **`benchmark.py` の実経路**に通す。

`tests/test_known_answer.py` の K1 は `metrics` / `stats` / 集計という評価器の中核を
通しているが、**読み込みと保存を通っていない**。R22 §5-1 が求めるのは

    読み込み → 摂動 → 評価 → 保存 → 集計 → 報告表

の全体である。**最後の表が、自己一貫性と GT 基準の区別を保っているか**を確かめる。

やり方
------
**手法を「安定した誤答を返す代用品」に差し替える**：

    T_0 = H G,    \\hat T(P) = H G P^{-1}       （H は水平 0.5 m ずらし）

代用品は `(src, dst, cfg)` しか受け取らないので、**入力から P を復元する**
（無摂動の source を保持しておき、Umeyama で P を厳密に求める）。

期待される表：
    selfconsistency_success_rate = 1.0     （自己基準では完璧）
    gt_success_rate              = 0.0     （GT 基準では全滅）
    gt_med_trans                ~= 0.5 m

**既定を汚さない**：代用品は `regbim/methods/experimental/`（gitignore 対象）に
一時的に置き、`--methods` で指定して実行し、**終わったら消す**。

    conda activate sni-slam
    python Registration/scripts/k1_end_to_end.py
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "Registration"))
os.chdir(REPO)

STUB = '''"""K1 の代用品：**安定した誤答**を返す（R22 §5-1）。

    T_hat(P) = H G P^{-1}

`T_0`（無摂動の出力）も `H G` になるので、**自己一貫性では完璧、GT 基準では 0.5 m ずれる。**

★ **代用品に GT を渡すのは、測定器を試験するための限定した対照である。**
  **性能評価で手法に GT を渡すことではない。**
  このモジュールは `experimental/`（gitignore 対象）にのみ置き、
  `--methods` で明示指定したときしか動かない。
"""

import json

import numpy as np

from ..base import BaseRegistration
from ... import metrics
from ...scale import umeyama
from .. import register_method

_ORIG = {}


@register_method("known_answer_stub")
class KnownAnswerStub(BaseRegistration):
    def register(self, src, dst, cfg):
        G = np.asarray(json.load(open(cfg["eval"]["t_gt_path"]))["T_gt"],
                       dtype=np.float64).reshape(4, 4)
        key = cfg["eval"]["t_gt_path"]
        if key not in _ORIG:
            # 最初の呼び出しは無摂動（benchmark は T0 を先に求める）
            _ORIG[key] = np.array(src.points, dtype=np.float64, copy=True)
            P_inv = np.eye(4)
        else:
            # 入力から摂動 P を厳密に復元する（transform_cloud がそのまま P を適用している）
            R, t, s = umeyama(_ORIG[key], np.asarray(src.points, dtype=np.float64))
            P_inv = metrics.invert_sim3(metrics.make_sim3(R, t, s))
        H = np.eye(4)
        H[0, 3] = 0.5                      # BIM 座標で水平に 0.5 m ずらす
        return H @ G @ P_inv
'''


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="Registration/configs/realdata/m3_block_b__E3.yaml")
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--out-dir", default="Registration/output/diag/k1_end_to_end")
    args = ap.parse_args()

    exp = os.path.join(REPO, "Registration", "regbim", "methods", "experimental")
    os.makedirs(exp, exist_ok=True)
    init_p = os.path.join(exp, "__init__.py")
    stub_p = os.path.join(exp, "known_answer_stub.py")
    made_init = not os.path.exists(init_p)
    if made_init:
        open(init_p, "w").write("")
    open(stub_p, "w").write(STUB)
    try:
        cmd = [sys.executable, "-W", "ignore", "Registration/scripts/benchmark.py",
               "--config", args.config, "--trials", str(args.trials),
               "--methods", "known_answer_stub", "--out-dir", args.out_dir]
        print("実行:", " ".join(cmd), flush=True)
        r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
        print(r.stdout[-3000:])
        if r.returncode != 0:
            print(r.stderr[-3000:])
            return r.returncode
    finally:
        os.remove(stub_p)
        if made_init:
            os.remove(init_p)
        shutil.rmtree(os.path.join(exp, "__pycache__"), ignore_errors=True)

    # ---- 報告表まで区別が保たれているかを確かめる ----
    print("\n=== 保存された表の検査（ここが K1 の本体）===")
    ok = True
    tp = os.path.join(args.out_dir, "trials.csv")
    rows = list(csv.DictReader(open(tp)))
    print("trials.csv: %d 行 / 列: %s" % (len(rows), ", ".join(rows[0].keys())))

    def col(name):
        return [r[name] for r in rows] if name in rows[0] else None

    sc = col("selfconsistency_success")
    gt = col("gt_success")
    if sc is None or gt is None:
        print("  **FAIL** selfconsistency_success / gt_success の列が無い")
        return 1
    n_sc = sum(v == "True" for v in sc)
    n_gt = sum(v == "True" for v in gt)
    print("  自己一貫性の成功 %d/%d / GT 基準の成功 %d/%d" % (n_sc, len(sc), n_gt, len(gt)))
    for name, cond, detail in [
            ("自己一貫性は全件合格", n_sc == len(sc), "%d/%d" % (n_sc, len(sc))),
            ("**GT 基準は全件不合格**", n_gt == 0, "%d/%d" % (n_gt, len(gt))),
            ("**両者が同じ値になっていない**", n_sc != n_gt, ""),
    ]:
        print(("  OK   " if cond else "  FAIL ") + name + ("  " + detail if detail else ""))
        ok &= bool(cond)

    trans = [float(r["gt_trans"]) for r in rows]
    print("  gt_trans: 中央 %.4f m（期待 0.5 付近）" % sorted(trans)[len(trans) // 2])

    sp = os.path.join(args.out_dir, "summary.json")
    if os.path.exists(sp):
        blob = json.load(open(sp))
        per = blob["methods"]
        entry = per[list(per.keys())[0]] if isinstance(per, dict) else per[0]
        r0 = entry.get("aggregate", entry)
        print("\nsummary.json の報告表:")
        for k in ("selfconsistency_success_rate", "gt_success_rate",
                  "selfconsistency_med_trans", "gt_med_trans", "n_degenerate"):
            if k in r0:
                print("   %-32s %s" % (k, r0[k]))
        fb = entry.get("gt_failure_breakdown", {})
        if fb:
            print("   gt_failure_breakdown: 失敗 %d 件 / 内訳 %s"
                  % (fb.get("n_failed"), fb.get("combinations")))
            by = (fb.get("by_criterion") or {}).get("trans") or {}
            print("   並進の超過量: 中央 %s m（期待 0.5 - 0.1 = 0.4）"
                  % by.get("median_excess"))
        for name, cond in [
                ("**報告表で selfconsistency_success_rate = 1.0**",
                 abs(float(r0.get("selfconsistency_success_rate", -1)) - 1.0) < 1e-9),
                ("**報告表で gt_success_rate = 0.0**",
                 abs(float(r0.get("gt_success_rate", -1)) - 0.0) < 1e-9)]:
            print(("  OK   " if cond else "  FAIL ") + name)
            ok &= bool(cond)
    else:
        print("  summary.json が無い（保存経路が通っていない）")
        ok = False

    print("\n" + ("all checks passed" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
