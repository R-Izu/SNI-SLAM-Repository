"""R49 段階1 の実行（sni-slam の環境、ブランチ `exp/r49-timing`）。**回す前に commit する。** 指示書 §3・§4・§8。

データ（target）
- synth：GT-A 8 シーン。config `gt_a_plan_a/<scene>.yaml`（R45 と同じ、参照 seed 0）、P = R45 の plan_a の trial 0〜99
- bim：BIM 由来の対照 E2（411）・E3（410＋411）。config `r39_bim/bim_<E>_plan_correlate.yaml`、P = R39 の trial 0〜99
- real：`m3_cor_a〜d` × 参照 E2・E3。config `realdata/<scene>__<E>.yaml` に `proposed.translation_init: plan_correlate`（R46 の planA と同じ）、
  source は slam_mesh の標本化 seed 0（R37・R46 と同じ。家具は background のまま）、P = R37 の P 系列 k 0〜9
方式
- full：`proposed`（案A）／single：`proposed_no_semantic`（ICP と最終の採点でクラスを区別しない）
- nowall：`proposed_no_wall_yaw`（壁の向きからのヨーの候補なし。重力は残し 0° から 10° おきの 36 向き。synth と real だけ）
- admit：`proposed` に `proposed.label_filter.enabled: false`（非構造物を入れる。記述）
- nograv：`proposed_no_gravity`（重心の初期化と自由回転の ICP。real だけ、記述。synth は `output/Registration/gt_a/` の既存の結果を使う）
- 4 方式（real だけ）：`baseline_open3d`（FPFH＋RANSAC、点対点）・`baseline_ransac_p2l`・`baseline_fgr`・`baseline_fgr_p2l`。
  `gt_scale` 属性を持つ方式は、benchmark と同じく P の縮尺を外して渡す
- 案A 系（full・single・nowall・admit）は `diagnostics: {record_yaw: true, record_stages: true}` を付ける（R32 §3 の段の分類のため。記録だけで計算は変わらない）
実行のしかた
- `serial`：full・single を 1 本ずつ（試行ごとに full → single の順、データ・シーン・試行の順）。`regbim.instrument` で段の時間と ICP の反復を記録する。
  スレッドは呼び出し側で固定（`taskset -c 0-7`、OMP・OpenBLAS・MKL = 8）。読み込みはシーンごとに 1 回で、その時間は別に記録する
- `parallel`：それ以外を並列（時間は使わない）。nowall は S6 のため synth の各シーン先頭 10 試行（80）を先に回せる（`--s6`）
出力：`~/rizu/r49_data/results/<data>/<method>/<target>.json`（"k" → T_est・P・time_s・instr）。途中から再開できる
使い方：python r49_run.py serial | parallel [並列数] [--s6]
"""
import copy, json, os, sys, time
from multiprocessing import Pool
import numpy as np
import yaml

REPO = os.path.expanduser("~/rizu/SNI-SLAM")
sys.path.insert(0, os.path.join(REPO, "Registration")); sys.path.insert(0, os.path.join(REPO, "Registration", "scripts"))
os.chdir(REPO)
from regbim import io_utils, metrics, instrument               # noqa: E402
from regbim.methods import get_method                          # noqa: E402
from failure_decomposition import provenance                   # noqa: E402

OUT = os.path.expanduser("~/rizu/r49_data/results")
SYNTH = ["room_0", "room_1", "room_2", "office_0", "office_1", "office_2", "office_3", "office_4"]
BIM = ["E2", "E3"]
REAL = ["m3_cor_%s__%s" % (s, e) for s in "abcd" for e in ("E2", "E3")]
METHOD = {"full": "proposed", "single": "proposed_no_semantic", "nowall": "proposed_no_wall_yaw", "admit": "proposed",
          "nograv": "proposed_no_gravity", "fpfh_p2p": "baseline_open3d", "fpfh_p2l": "baseline_ransac_p2l",
          "fgr_p2p": "baseline_fgr", "fgr_p2l": "baseline_fgr_p2l"}
PLAN_A = ("full", "single", "nowall", "admit")


def targets(data):
    return {"synth": SYNTH, "bim": BIM, "real": REAL}[data]


def n_trials(data):
    return 10 if data == "real" else 100


def config(data, target, m):
    if data == "synth":
        cfg = yaml.safe_load(open("Registration/configs/gt_a_plan_a/%s.yaml" % target))
    elif data == "bim":
        cfg = yaml.safe_load(open("Registration/configs/r39_bim/bim_%s_plan_correlate.yaml" % target))
    else:
        cfg = yaml.safe_load(open("Registration/configs/realdata/%s.yaml" % target))
        cfg["source"] = dict(cfg["source"], seed=0)
        cfg.setdefault("proposed", {})["translation_init"] = "plan_correlate"
    cfg = copy.deepcopy(cfg)
    if m == "admit":
        cfg.setdefault("proposed", {})["label_filter"] = {"enabled": False}
    if m in PLAN_A:
        cfg["diagnostics"] = {"record_yaw": True, "record_stages": True}
    return cfg


def perturbations(data, target):
    if data == "synth":
        tm = json.load(open("output/Registration/r45/plan_a/%s/trial_matrices.json" % target))
        rows = {t["trial"]: t for t in tm["trials"] if t["method"] == "proposed" and t["trial"] >= 0}
    elif data == "bim":
        tm = json.load(open("output/Registration/r39_bim/%s_plan_correlate/trial_matrices.json" % target))
        rows = {t["trial"]: t for t in tm["trials"] if t["method"] == "proposed" and t["trial"] >= 0}
    else:
        rows = {r["k"]: r for r in json.load(open("Registration/output/diag/r37_replicate.json"))["rows"]
                if r["series"] == "P" and r["target"] == target}
    return [np.asarray(rows[k]["P"], dtype=np.float64) for k in range(n_trials(data))]


_CACHE = {}


def clouds(data, target):
    key = (data, target)
    if key not in _CACHE:
        cfg = config(data, target, "full")
        t0 = time.perf_counter()
        src = io_utils.load_source_cloud(cfg); dst = io_utils.load_reference_cloud(cfg)
        _CACHE.clear(); _CACHE[key] = (src, dst, time.perf_counter() - t0)
    return _CACHE[key]


def compact(rec):
    """記録を小さくする：短い精緻化は呼び出しごとの反復・止まり方・時間だけ、通常・段2・自由回転は反復ごとの値も残す"""
    calls = []
    for c in rec["icp"]:
        e = {k: c[k] for k in ("tag", "iters", "stop", "max_iter", "t_tree", "t_total")}
        if c["tag"] != "icp_short":
            e.update(deltas=c["deltas"], n_corr=c["n_corr"], resid=c["resid"])
        calls.append(e)
    return {"stages": rec["stages"], "icp": calls, "winner_index": rec.get("winner_index")}


def run_one(data, target, m, k, P, timed):
    src, dst, t_read = clouds(data, target)
    cfg = config(data, target, m); meth = get_method(METHOD[m])
    if getattr(meth, "gt_scale", False):
        R, t, s = metrics.decompose_sim3(P); P = metrics.make_sim3(R, t / s, 1.0)
    if timed:
        instrument.start()
    t0 = time.perf_counter()
    T = meth.register(metrics.transform_cloud(src, P), dst, cfg)
    dt = time.perf_counter() - t0
    rec = compact(instrument.stop()) if timed else None
    diag = None
    if m in PLAN_A:                       # 段の分類（R32 §3）のため：相関の候補・短い精緻化の後・通常の精緻化の後の変換
        pd, yd = meth.last_plan_diag, meth.last_yaw_diag
        diag = {"gen": [c["T_gen"] for c in pd["stage_candidates"]], "short": [c["T_short"] for c in pd["stage_candidates"]],
                "after_icp": yd["candidate_T"], "scores": yd["candidate_scores"]}
    return {"T_est": np.asarray(T, dtype=np.float64).tolist(), "P": P.tolist(), "time_s": dt, "read_s": t_read, "instr": rec, "diag": diag}


def path(data, m, target):
    return os.path.join(OUT, data, m, target + ".json")


def load(data, m, target):
    p = path(data, m, target)
    return json.load(open(p)) if os.path.exists(p) else {"provenance": provenance(), "rows": {}}


def save(data, m, target, d):
    os.makedirs(os.path.dirname(path(data, m, target)), exist_ok=True)
    json.dump(d, open(path(data, m, target), "w"))


def serial():
    env = {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
    env["affinity"] = sorted(os.sched_getaffinity(0)); env["cpu_count"] = os.cpu_count()
    json.dump(env, open(os.path.join(OUT, "serial_env.json"), "w"))
    for data in ("synth", "bim", "real"):
        for target in targets(data):
            Ps = perturbations(data, target)
            done = {m: load(data, m, target) for m in ("full", "single")}
            for k in range(n_trials(data)):
                for m in ("full", "single"):
                    if str(k) in done[m]["rows"]:
                        continue
                    done[m]["rows"][str(k)] = run_one(data, target, m, k, Ps[k], True)
                    save(data, m, target, done[m])
                    print("serial", data, target, m, k, "%.1fs" % done[m]["rows"][str(k)]["time_s"], flush=True)


def pjob(a):
    data, target, m, k = a
    return a, run_one(data, target, m, k, perturbations(data, target)[k], False)


def parallel(npar, s6):
    if s6:
        jobs = [("synth", t, "nowall", k) for t in SYNTH for k in range(10)]
    else:
        jobs = [("synth", t, "nowall", k) for t in SYNTH for k in range(100)] + [("real", t, "nowall", k) for t in REAL for k in range(10)]
        jobs += [(d, t, "admit", k) for d in ("synth", "bim", "real") for t in targets(d) for k in range(n_trials(d))]
        jobs += [("real", t, m, k) for t in REAL for m in ("nograv", "fpfh_p2p", "fpfh_p2l", "fgr_p2p", "fgr_p2l") for k in range(10)]
    have = {}
    todo = []
    for a in jobs:
        key = (a[0], a[2], a[1])
        if key not in have:
            have[key] = load(a[0], a[2], a[1])
        if str(a[3]) not in have[key]["rows"]:
            todo.append(a)
    todo.sort(key=lambda a: (a[0], a[1]))        # 同じ target を続けて、読み込みを使い回す
    print("todo", len(todo), flush=True)
    t0 = time.time()
    with Pool(npar) as pool:
        for i, (a, r) in enumerate(pool.imap_unordered(pjob, todo, chunksize=1)):
            key = (a[0], a[2], a[1]); have[key]["rows"][str(a[3])] = r; save(a[0], a[2], a[1], have[key])
            print("parallel", i + 1, len(todo), a, "%.0fs" % r["time_s"], "elapsed %.0fs" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    if sys.argv[1] == "serial":
        serial()
    else:
        parallel(int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 8, "--s6" in sys.argv)
