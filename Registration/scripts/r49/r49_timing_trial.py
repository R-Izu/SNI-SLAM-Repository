"""R49 段階0 §1-4：時間の測り方の試し（sni-slam の環境、1 プロセスで順に）。GT-A room_0 × 先頭 5 試行 × 3 回。

- 実行：`taskset -c 0-7`、OMP_NUM_THREADS = OPENBLAS_NUM_THREADS = MKL_NUM_THREADS = 8（呼び出し側で固定）。
  scipy の cKDTree の query は `workers=-1`（手法のコード）なので、スレッドは os.cpu_count() 本作られ、taskset の 8 コアの上で動く
- 最初に 1 回（trial 0）回して捨てる（読み込みの温まり）
- 段：read（config から source・参照を読む。register の外）、prep・rotation・correlate・icp・select（`regbim.instrument`）、total（register の全体）
出力：`Registration/output/diag/r49/timing_trial.json`
"""
import json, os, sys, time
import numpy as np
import yaml

REPO = os.path.expanduser("~/rizu/SNI-SLAM")
sys.path.insert(0, os.path.join(REPO, "Registration")); os.chdir(REPO)
from regbim import io_utils, metrics, instrument                 # noqa: E402
from regbim.methods import get_method                            # noqa: E402

SCENE = "room_0"; TRIALS = range(5); REPS = 3


def one(cfg, row):
    t0 = time.perf_counter()
    src = io_utils.load_source_cloud(cfg); dst = io_utils.load_reference_cloud(cfg)
    t_read = time.perf_counter() - t0
    P = np.asarray(row["P"], dtype=np.float64); s = metrics.transform_cloud(src, P)
    instrument.start(); t1 = time.perf_counter()
    T = get_method("proposed").register(s, dst, cfg)
    t_total = time.perf_counter() - t1; rec = instrument.stop()
    icp = {}
    for c in rec["icp"]:
        e = icp.setdefault(c["tag"], {"calls": 0, "iters": 0, "time": 0.0, "stop": {}})
        e["calls"] += 1; e["iters"] += c["iters"]; e["time"] += c["t_total"]; e["stop"][c["stop"]] = e["stop"].get(c["stop"], 0) + 1
    return {"read": t_read, "total": t_total, "stages": rec["stages"], "icp": icp,
            "bit_equal_r45": bool(np.array_equal(np.asarray(T), np.asarray(row["T_est"])))}


def main():
    cfg = yaml.safe_load(open("Registration/configs/gt_a_plan_a/%s.yaml" % SCENE))
    tm = json.load(open("output/Registration/r45/plan_a/%s/trial_matrices.json" % SCENE))
    rows = {t["trial"]: t for t in tm["trials"] if t["trial"] >= 0}
    env = {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
    env["affinity"] = sorted(os.sched_getaffinity(0)); env["cpu_count"] = os.cpu_count()
    warm = one(cfg, rows[0])
    out = {"scene": SCENE, "env": env, "warmup_discarded": warm, "runs": []}
    for k in TRIALS:
        for r in range(REPS):
            x = one(cfg, rows[k]); x.update(trial=k, rep=r); out["runs"].append(x)
            print(k, r, "total %.2f s" % x["total"], {a: round(b, 2) for a, b in x["stages"].items()}, "read %.2f" % x["read"], flush=True)
    os.makedirs("Registration/output/diag/r49", exist_ok=True)
    json.dump(out, open("Registration/output/diag/r49/timing_trial.json", "w"), indent=1)


if __name__ == "__main__":
    main()
