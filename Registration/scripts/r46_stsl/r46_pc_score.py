"""R46 §8-1 陽性対照の判定。**結果を見る前に commit する。**

合格の線（main が §8-1 で固定）：10 試行中 8 試行以上が、STSL 自身の成功判定（並進 3 m 未満・ヨー 5° 未満）を満たす。
前処理は K = 5 回（§8-5）。**試行ごとの成否は 5 回の多数決（3 回以上の成功）**とする（§8-5 の多数決の規則を、ここでも使う）。
設定 A・B（§8-3）は両方出す。**どちらかの設定で合格すれば、陽性対照は合格**（§8-3 で STSL には良い方の設定を採るため）。
k ごとの件数（10 試行中）と、全 50 回の件数も記述として出す。

使い方：python r46_pc_score.py（libim でも sni-slam でもよい。numpy と json だけを使う）
"""
import json, os, re


def jl(line):
    """C++ が書く inf・nan を JSON の Infinity・NaN に直して読む"""
    return json.loads(re.sub(r"(?<![A-Za-z])(-?)inf(?![A-Za-z])", lambda m: m.group(1) + "Infinity", line).replace("nan", "NaN"))

ROOT = os.environ.get("R46_STSL_ROOT", os.path.expanduser("~/rizu/r46_stsl_data"))
K, N, FLOOR = 5, 10, "M3_410_411"


def main():
    res = {"rule": "majority (>=3 of 5) per trial; pass if >=8 of 10 in at least one setting", "settings": {}}
    for st in ("A", "B"):
        per = {i: [] for i in range(N)}; by_k = {}
        for k in range(1, K + 1):
            rows = [jl(l) for l in open(os.path.join(ROOT, "results", "pc", "%s_k%d_%s.jsonl" % (st, k, FLOOR)))]
            assert len(rows) == N, (st, k, len(rows))
            by_k[k] = sum(r["success_stsl"] for r in rows)
            for r in rows:
                per[r["index"]].append({"ok": r["success_stsl"], "n_cand": len(r["candidates"]),
                                        "n_correct": sum(c["correct"] for c in r["candidates"]),
                                        "score": r["optimal_score"], "gt_score": r["gt_score"], "reg_s": r["reg_s"]})
        maj = {i: sum(x["ok"] for x in v) >= 3 for i, v in per.items()}
        n_maj = sum(maj.values())
        res["settings"][st] = {"n_success_majority": n_maj, "pass": n_maj >= 8, "by_k": by_k,
                               "n_success_all_runs": sum(by_k.values()), "per_trial": {str(i): v for i, v in per.items()}}
        print(st, "majority %d/10 pass=%s | by k %s | all runs %d/50" % (n_maj, n_maj >= 8, by_k, sum(by_k.values())))
    res["pass"] = any(v["pass"] for v in res["settings"].values())
    print("positive control pass:", res["pass"])
    json.dump(res, open(os.path.join(ROOT, "results", "pc", "pc_score.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
