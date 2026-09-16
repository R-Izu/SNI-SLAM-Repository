"""R27 §3 — GT-A 基準の置き換え表を作る（8 シーン × 8 手法 × 100 試行）。

**旧値を消さない。並べる**（R27 §4）。
**母数を出す**（成功数・試行数）。**率だけを返さない。**
**自己一貫性と GT 基準を両方出す**（K1 で通した区別をこの表でも保つ）。
**シーン別に出す。平均だけにしない。**
**Wilson CI をつける。**

旧値の基準はシーンで違う（R26 §1）ので、**シーンごとに注記する**：
`room_0` だけ凍結 T_gt（＝提案手法自身の解）があり、他 7 シーンは無かった。

    python Registration/scripts/gt_a_table.py
"""

from __future__ import annotations

import csv
import json
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "Registration"))
sys.path.insert(0, os.path.join(REPO, "Registration", "scripts"))
os.chdir(REPO)

from failure_decomposition import provenance     # noqa: E402
from regbim.stats import wilson_ci               # noqa: E402

SCENES = ["room_0", "room_1", "room_2", "office_0", "office_1",
          "office_2", "office_3", "office_4"]
METHODS = ["proposed", "proposed_no_semantic", "proposed_no_gravity",
           "proposed_fixed_scale", "baseline_open3d", "baseline_ransac_p2l",
           "baseline_fgr", "baseline_fgr_p2l"]
NEW = "output/Registration/gt_a"
OLD = "output/Registration/sectionC"


def read(base: str, scene: str):
    p = os.path.join(base, scene, "results.csv")
    if not os.path.exists(p):
        return {}
    return {r["method"]: r for r in csv.DictReader(open(p))}


def rate(row, key):
    """(成功数, 試行数) を返す。**率ではなく母数で持つ**（合算のため）。

    ★ 自己一貫性の列名は新旧で違う。R7 §3 で `success_rate` →
      `selfconsistency_success_rate` に改名された。**旧 sectionC の出力は改名前である。**
      どちらも「各手法が自分の無摂動解に戻れたか」であり、**同じ量の別名**なので
      両方を受ける。GT 基準の `gt_success_rate` は新しい列で、旧出力には存在しない。
    """
    if not row:
        return None
    names = ([key] if key != "selfconsistency_success_rate"
             else [key, "success_rate"])
    for k in names:
        if row.get(k, "") not in ("", None):
            n = int(float(row["robust_trials"]))
            return int(round(float(row[k]) * n)), n
    return None


def main() -> int:
    new = {s: read(NEW, s) for s in SCENES}
    old = {s: read(OLD, s) for s in SCENES}
    missing = [s for s in SCENES if not new[s]]
    if missing:
        print("**未完のシーン: %s**" % ", ".join(missing))

    # 旧値の基準（R26 §1：凍結 T_gt があったのは room_0 だけ）
    basis = {}
    for s in SCENES:
        log = os.path.join(OLD, "%s_benchmark.log" % s)
        frozen = os.path.exists(log) and "no frozen T_gt" not in open(log).read()
        basis[s] = "旧T_gt(提案手法の解)" if frozen else "各手法 自分の解"

    print("## 旧評価の基準（シーン別。同じ数字が別の量である）\n")
    for s in SCENES:
        print("  %-10s %s" % (s, basis[s]))

    rows = []
    for key, title in (("gt_success_rate", "GT 基準（新・GT-A）"),
                       ("selfconsistency_success_rate", "自己一貫性")):
        print("\n\n## %s\n" % title)
        print("%-22s %s %12s" % ("手法", "".join("%-8s" % s.replace("office_", "of")
                                                 .replace("room_", "rm")
                                                 for s in SCENES), "合計"))
        print("-" * (23 + 8 * len(SCENES) + 13))
        for m in METHODS:
            cells, K, N = [], 0, 0
            for s in SCENES:
                r = rate(new[s].get(m), key)
                if r is None:
                    cells.append("%-8s" % "—")
                else:
                    k, n = r
                    cells.append("%-8.2f" % (k / n))
                    K += k
                    N += n
            if N:
                lo, hi = wilson_ci(K, N)
                tot = "%3d/%3d=%.1f%% [%.1f,%.1f]" % (K, N, 100 * K / N,
                                                      100 * lo, 100 * hi)
                rows.append({"metric": key, "method": m, "k": K, "n": N,
                             "rate": K / N, "ci_lo": lo, "ci_hi": hi})
            else:
                tot = "—"
            print("%-22s %s %s" % (m, "".join(cells), tot))

    print("\n\n## 旧値（sectionC）との対照 —— **旧値を消さない**\n")
    print("%-22s %22s %28s" % ("手法", "旧・自己一貫性", "新・GT 基準（GT-A）"))
    print("-" * 74)
    for m in METHODS:
        oK = oN = 0
        for s in SCENES:
            r = rate(old[s].get(m), "selfconsistency_success_rate")
            if r:
                oK += r[0]
                oN += r[1]
        nr = [x for x in rows if x["metric"] == "gt_success_rate" and x["method"] == m]
        o = "%3d/%3d = %.1f%%" % (oK, oN, 100 * oK / oN) if oN else "（未実施）"
        n = ("%3d/%3d = %.1f%%  [%.1f, %.1f]"
             % (nr[0]["k"], nr[0]["n"], 100 * nr[0]["rate"],
                100 * nr[0]["ci_lo"], 100 * nr[0]["ci_hi"])) if nr else "—"
        print("%-22s %22s %28s" % (m, o, n))

    print("\n**旧値は 8 シーン中 7 シーンで『各手法が自分の無摂動解に戻れたか』であり、"
          "GT 基準ではない**（R26 §1）。**同じ列に見えても別の量である。**")

    out = "Registration/output/diag/gt_a_table.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"provenance": provenance(), "basis_old": basis, "rows": rows},
                  f, indent=2, ensure_ascii=False)
    print("\nwrote %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
