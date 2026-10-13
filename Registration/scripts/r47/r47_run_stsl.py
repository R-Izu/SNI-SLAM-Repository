"""R47 の STSL 側（libim の環境）。**回す前に commit する。STSL のパラメータ・コードは変えない。**

B-1 の交差の組：R46 で前処理済みの入力（`processed/real_<input_set>_<設定>_k<k>/30m/`）を、もう一方の参照の BIM で登録し直す
  （室内面・両面 × 設定 A・B、K = 5。STSL（主）= 両面・設定 A、副 = 4 変種）
B-2：入力 = Rz(ψ) · G1 · p（BIM 座標、z が上、並進 0）、ψ = 0〜350°（10° ごと）。正解 T_true（入力 → BIM）= Rz(−ψ)。
  STSL（主）= 両面・設定 A だけ、K = 5。参照 410＋411（`M3_410_411_both`）
使い方：python r47_run_stsl.py [並列数]
"""
import json, os, random, sys, time
import multiprocessing as mp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "r46_stsl"))
for _try in range(20):
    try:
        import r46_stsl_pipeline as pl                                  # noqa: E402
        break
    except FileExistsError:
        for m in [m for m in sys.modules if m.startswith("preprocess") or m == "r46_stsl_pipeline"]:
            del sys.modules[m]
        time.sleep(1.0 + random.random())
from r47_common import b1_conditions, B2_INPUTS, YAWS, FLOOR          # noqa: E402

ROOT = pl.ROOT
SNI = os.path.expanduser("~/rizu/SNI-SLAM")


def job_b1(a):
    """1 つの入力の組について、参照の BIM（室内面・両面）× 設定 A・B を順に回す（pose.txt を同時に書かないため）"""
    input_set, floors = a
    setname = "real_" + input_set
    spec = json.load(open(os.path.join(ROOT, "inputs", setname, "spec.json")))
    T_trues = [np.asarray(t["T_true"], dtype=np.float64) for t in spec["trials"]]
    for floor, st, k in [(f, st, k) for f in floors for st in "AB" for k in range(1, 6)]:
        if os.path.exists(os.path.join(ROOT, "results", setname, "%s_k%d_%s.jsonl" % (st, k, floor))):
            continue
        seq = "%s_%s_k%d" % (setname, st, k); seq_dir = os.path.join(ROOT, "processed", seq, "30m")
        trp = {int(i): v for i, v in json.load(open(os.path.join(seq_dir, "T_rp.json"))).items()}
        pl.register(setname, st, k, seq, seq_dir, T_trues, trp, floor)
    return a


def make_b2_inputs():
    sp = json.load(open(os.path.join(SNI, "Registration/output/diag/criterion_spread.json")))["scenes"]
    for s, cut in B2_INPUTS:
        name = "r47rot_%s_%s" % (s, cut); d = os.path.join(ROOT, "inputs", name)
        if os.path.exists(os.path.join(d, "spec.json")):
            continue
        os.makedirs(d, exist_ok=True)
        G1 = np.asarray([r["T"] for r in sp[s]["results"] if int(r["id"]) == 1][0], dtype=np.float64)
        p = np.load(os.path.join(ROOT, "inputs_sni", "%s__E3_r%s.npz" % (s, cut)))["points"]
        q0 = p @ G1[:3, :3].T + G1[:3, 3]
        spec = {"scene": s, "cut": cut, "trials": []}
        for i, y in enumerate(YAWS):
            c, sn = np.cos(np.radians(y)), np.sin(np.radians(y))
            Rz = np.eye(4); Rz[:3, :3] = [[c, -sn, 0], [sn, c, 0], [0, 0, 1]]
            np.save(os.path.join(d, "%06d.npy" % i), (q0 @ Rz[:3, :3].T).astype(np.float32))
            spec["trials"].append({"index": i, "yaw_deg": float(y), "T_true": np.linalg.inv(Rz).tolist(), "P_src_to_input": (Rz @ G1).tolist()})
        json.dump(spec, open(os.path.join(d, "spec.json"), "w"))


def job_b2(a):
    s, cut = a
    name = "r47rot_%s_%s" % (s, cut)
    for k in range(1, 6):
        if os.path.exists(os.path.join(ROOT, "results", name, "A_k%d_M3_410_411_both.jsonl" % k)):
            continue
        seq, seq_dir, T_trues, trp = pl.prep(name, "A", k)
        pl.register(name, "A", k, seq, seq_dir, T_trues, trp, "M3_410_411_both")
    return a


def main():
    npar = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    make_b2_inputs()
    b1 = [(c["input_set"], (FLOOR[c["ref"]], FLOOR[c["ref"]] + "_both")) for c in b1_conditions() if c["cross"]]
    with mp.get_context("spawn").Pool(npar) as p:
        for a in p.imap_unordered(job_b1, b1):
            print("b1 done", a, flush=True)
        for a in p.imap_unordered(job_b2, B2_INPUTS):
            print("b2 done", a, flush=True)
    print("all done")


if __name__ == "__main__":
    main()
