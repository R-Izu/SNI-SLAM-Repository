"""R46 段階3：STSL に縮尺の探索を足して回す（libim の環境）。**回す前に commit する。STSL のコードは変えない。**

縮尺の格子（§10-3）：σ_j = exp(linspace(log 0.3, log 3, 59))、j = 0..58（案A と同じ J = 59、[0.3, 3] の対数格子）。
各 (set, 設定 A・B, j, k = 1..5) で、入力 q を σ_j 倍した点で STSL の公式の前処理と登録を回す（`r46_stsl_pipeline` の prep・register）。
最良の縮尺の選び方（採点側 `r46_s3_score.py`）：STSL 自身の検証スコア（optimal_score）が最大の縮尺。候補なしの縮尺は飛ばし、
全縮尺で候補なしなら無効。
容量のため、縮尺をかけた入力と前処理の中間ファイルは登録の後に消す（T_rp と結果の jsonl は残す）。
使い方：python r46_s3_run.py [並列数] [set の接頭辞 ...]（例：bim_ で BIM 由来の対照だけ）
"""
import json, os, shutil, sys
from multiprocessing import Pool
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r46_stsl_pipeline as pl                                          # noqa: E402

ROOT = pl.ROOT
SIGMAS = np.exp(np.linspace(np.log(0.3), np.log(3.0), 59))
SETS = {"gta_%s" % s: "GTA_%s" % s for s in ["room_0", "room_1", "room_2", "office_0", "office_1", "office_2", "office_3", "office_4"]}
SETS.update({"bim_E2": "M3_411_both", "bim_E3": "M3_410_411_both"})
K = 5


def job(a):
    base, st, j = a
    name = "s3_%s_j%02d" % (base, j)
    src = os.path.join(ROOT, "inputs_s3", base); tmp = os.path.join(ROOT, "inputs", name)
    os.makedirs(tmp, exist_ok=True)
    spec = json.load(open(os.path.join(src, "spec.json")))
    D = np.diag([SIGMAS[j]] * 3 + [1.0]); Di = np.linalg.inv(D)
    for t in spec["trials"]:
        np.save(os.path.join(tmp, "%06d.npy" % t["index"]), (np.load(os.path.join(src, "%06d.npy" % t["index"])) * SIGMAS[j]).astype(np.float32))
        t["T_true"] = (np.asarray(t["T_true"]) @ Di).tolist()
    spec["sigma"] = float(SIGMAS[j]); json.dump(spec, open(os.path.join(tmp, "spec.json"), "w"))
    res_dir = os.path.join(ROOT, "results", name); os.makedirs(res_dir, exist_ok=True)
    for k in range(1, K + 1):
        if os.path.exists(os.path.join(res_dir, "%s_k%d_%s.jsonl" % (st, k, SETS[base]))):
            continue
        seq, seq_dir, T_trues, trp = pl.prep(name, st, k)
        pl.register(name, st, k, seq, seq_dir, T_trues, trp, SETS[base])
        shutil.copy(os.path.join(seq_dir, "T_rp.json"), os.path.join(res_dir, "%s_k%d_T_rp.json" % (st, k)))
        shutil.rmtree(os.path.join(ROOT, seq), ignore_errors=True)
        shutil.rmtree(os.path.join(ROOT, "processed", seq), ignore_errors=True)
        for f in ("_pointsam.yaml", "_interval_prep.yaml"):
            p = os.path.join(ROOT, "cfg", seq + f)
            if os.path.exists(p):
                os.remove(p)
    shutil.rmtree(tmp, ignore_errors=True)
    return a


def main():
    npar = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    pref = sys.argv[2:] or [""]
    jobs = [(b, st, j) for b in SETS if any(b.startswith(x) for x in pref) for st in "AB" for j in range(len(SIGMAS))]
    first = {}
    for a in jobs:
        first.setdefault((SETS[a[0]], a[1]), a)
    for a in first.values():                          # ハッシュ表を作る最初の 1 組は順に
        print("warm", job(a), flush=True)
    rest = [a for a in jobs if a not in first.values()]
    with Pool(npar) as p:
        for i, a in enumerate(p.imap_unordered(job, rest)):
            if i % 20 == 0:
                print("done", i + 1, "/", len(rest), a, flush=True)
    print("all done")


if __name__ == "__main__":
    main()
