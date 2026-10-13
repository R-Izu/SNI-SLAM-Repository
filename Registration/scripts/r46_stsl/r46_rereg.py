"""R46 §10-2 B1：BIM 入力（両面）で STSL の登録だけを回し直す（libim の環境）。**source 側の前処理は段階1のものを再利用する。**

段階1の 32 条件 × 設定 A・B × K = 5 の前処理（`processed/<seq>/30m/`、T_rp.json）をそのまま使い、
BIM を `<参照>_both`（`r46_stsl_bim.py both`）にして `r46_stsl_pipeline.register` を呼ぶ。
結果：`results/real_<target>_r<r>/<設定>_k<k>_<参照>_both.jsonl`
ハッシュ表の作成が並列で重ならないよう、各（参照・設定）の最初の 1 組は順に回し、残りを並列にする。
使い方：python r46_rereg.py [並列数] [変種 both|axis（既定 both）]
B3（§11-2）：変種 axis で、壁の基準線の BIM（`r46_stsl_bim_axis.py`）を使う。
"""
import json, os, sys
from multiprocessing import Pool
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r46_stsl_pipeline as pl                                   # noqa: E402  （STSL の位置へ chdir し、LiBIM_UST_ROOT を設定する）

ROOT = pl.ROOT
VARIANT = sys.argv[2] if len(sys.argv) > 2 else "both"
FLOOR = {"E2": "M3_411_" + VARIANT, "E3": "M3_410_411_" + VARIANT}
SETS = ["real_m3_cor_%s__%s_r%s" % (s, e, c) for s in "abcd" for e in ("E2", "E3") for c in ("0p5", "3", "8", "inf")]


def job(a):
    setname, st = a
    spec = json.load(open(os.path.join(ROOT, "inputs", setname, "spec.json")))
    T_trues = [np.asarray(t["T_true"], dtype=np.float64) for t in spec["trials"]]
    floor = FLOOR[setname.split("__")[1].split("_")[0]]
    out = []
    for k in range(1, 6):
        seq = "%s_%s_k%d" % (setname, st, k)
        seq_dir = os.path.join(ROOT, "processed", seq, "30m")
        trp = {int(i): v for i, v in json.load(open(os.path.join(seq_dir, "T_rp.json"))).items()}
        out.append(pl.register(setname, st, k, seq, seq_dir, T_trues, trp, floor))
    return a, out


def main():
    npar = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    jobs = [(s, st) for s in SETS for st in "AB"]
    first = {}
    for s, st in jobs:                                           # 各（参照・設定）の最初の 1 組
        key = (FLOOR[s.split("__")[1].split("_")[0]], st)
        first.setdefault(key, (s, st))
    for a in first.values():
        print("warm", job(a)[0], flush=True)
    rest = [a for a in jobs if a not in first.values()]
    with Pool(npar) as p:
        for a, _ in p.imap_unordered(job, rest):
            print("done", a, flush=True)


if __name__ == "__main__":
    main()
