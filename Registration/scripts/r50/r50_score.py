"""R50 段階1 A の採点（sni-slam の環境）。**回す前に commit する。** 指示書 §2-2・§4-1・§6。

- 完全：R49 の 1 本ずつの実行（`~/rizu/r49_data/results/<data>/full/`）。新しい 3 方式：`~/rizu/r50_data/results/`
- 採点・正解・大きな失敗（無効・e_R ≥ 45°・e_Ω ≥ 1.0 m）・段の分類（R32 §3）は R49 の `r49_score` と同じ関数。実データは G1、G2〜G8 は感度
- T1：synth、完全 対 採点だけクラスなし／T2：synth、完全 対 ICP だけクラスなし／T3：synth、完全 対 参照のクラスを幾何から作る／
  T4：real（G1）、完全 対 参照のクラスを幾何から作る。いずれも試行の対の正確 McNemar（両側）。水準 0.05/4 = 0.0125
- 表：方式 × データの大きな失敗、e_R・e_Ω の中央値・95%（有効な試行）、失敗した試行の段の分類の件数、シーンごとの失敗の数
出力：`Registration/output/diag/r50/r50_score.json`
"""
import json, os, sys
from collections import Counter
import numpy as np

REPO = os.path.expanduser("~/rizu/SNI-SLAM")
for p in ("Registration", "Registration/scripts", "Registration/scripts/r49", "Registration/scripts/r50"):
    sys.path.insert(0, os.path.join(REPO, p))
os.chdir(REPO)
from failure_decomposition import provenance                    # noqa: E402
from r41_rescore import per_trial                               # noqa: E402
import r49_run as RR                                            # noqa: E402
from r49_score import truth, stage_class, big, mcnemar, q       # noqa: E402

ALPHA = 0.05 / 4
R49 = os.path.expanduser("~/rizu/r49_data/results")
R50 = os.path.expanduser("~/rizu/r50_data/results")
SRC = {"full": R49, "score_only": R50, "icp_only": R50, "geo": R50}


def main():
    res = {"provenance": provenance(), "alpha": ALPHA}
    trial = {}
    for data in ("synth", "bim", "real"):
        for target in RR.targets(data):
            Gs, om, _ = truth(data, target)
            for m, base in SRC.items():
                rows = json.load(open(os.path.join(base, data, m, target + ".json")))["rows"]
                for k, r in rows.items():
                    per = {j: per_trial(r["T_est"], r["P"], G, om, None) for j, G in Gs.items()}
                    e = {"per": per["1"], "fail": {j: big(per[j]) for j in per}}
                    if e["fail"]["1"] and r.get("diag"):
                        e["stage"] = stage_class(r, Gs["1"], om)["class"]
                    trial[(data, m, target, int(k))] = e
    summ = {}
    for data in ("synth", "bim", "real"):
        for m in SRC:
            es = {(t, k): e for (d, mm, t, k), e in trial.items() if d == data and mm == m}
            v = [e["per"] for e in es.values() if e["per"]["v"]]
            fc = Counter()
            for e in es.values():
                for j, f in e["fail"].items():
                    fc[j] += int(f)
            summ["%s|%s" % (data, m)] = {
                "n": len(es), "n_fail_by_criterion": dict(fc), "eR": q([p["eR"] for p in v]), "eO": q([p["eO"] for p in v]),
                "n_invalid": sum(not e["per"]["v"] for e in es.values()),
                "fail_stage_class": dict(Counter(e["stage"] for e in es.values() if e.get("stage"))),
                "fail_per_target": {t: sum(es[(t, k)]["fail"]["1"] for k in range(RR.n_trials(data))) for t in RR.targets(data)}}
    res["summary"] = summ
    tests = {}
    for name, data, m in (("T1", "synth", "score_only"), ("T2", "synth", "icp_only"), ("T3", "synth", "geo"), ("T4", "real", "geo")):
        pairs = [(trial[(data, "full", t, k)]["fail"]["1"], trial[(data, m, t, k)]["fail"]["1"])
                 for t in RR.targets(data) for k in range(RR.n_trials(data))]
        n10 = sum(a and not b for a, b in pairs); n01 = sum(b and not a for a, b in pairs)
        tests[name] = {"data": data, "method": m, "n_pairs": len(pairs), "fail_full": sum(a for a, _ in pairs),
                       "fail_method": sum(b for _, b in pairs), "n10_full_only": n10, "n01_method_only": n01, "p": mcnemar(n10, n01), "alpha": ALPHA}
    res["tests"] = tests
    os.makedirs("Registration/output/diag/r50", exist_ok=True)
    json.dump(res, open("Registration/output/diag/r50/r50_score.json", "w"), ensure_ascii=False, default=float)
    for k, v in tests.items():
        print(k, v)


if __name__ == "__main__":
    main()
