"""R48 段階1：STSL を回す（libim の環境。STSL `bd23507`、コード・パラメータは変えない）。**回す前に commit する。**

- 設定 A（公式 `2f_office_02`、30 m）に固定、K = 5（平面分割は seed を固定できない）。前処理と登録は `r46_stsl_pipeline` の prep・register
- データの置き場所：`$R46_STSL_ROOT = ~/rizu/r48_data/`（inputs・processed・results・BIM）
- 本番：`inputs/<ref>__<cut>/`（31 参照 × 範囲、計 77 組 × 20 試行）を、BIM `BIM/<ref>/`（`lines_from_walls_v2`）で
- 陽性対照（記述）：`inputs/pc_<ref>/`（10 試行）を同じ BIM で
- 参照（BIM）ごとのハッシュ表は最初の登録で作られるので、参照ごとに最初の 1 組だけ先に回し（参照どうしは並列）、残りを並列にする
使い方：python r48_run_stsl.py [並列数]
"""
import json, os, random, shutil, sys, time
import multiprocessing as mp

os.environ["R46_STSL_ROOT"] = os.path.expanduser("~/rizu/r48_data/")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "r46_stsl"))
for _try in range(20):
    try:
        import r46_stsl_pipeline as pl                                  # noqa: E402
        break
    except FileExistsError:
        for m in [m for m in sys.modules if m.startswith("preprocess") or m == "r46_stsl_pipeline"]:
            del sys.modules[m]
        time.sleep(1.0 + random.random())
import r48_common as C                                                  # noqa: E402

ROOT = pl.ROOT
K = 5


def sets():
    out = []
    for ref in C.all_refs():
        out += [("%s__%s" % (ref, cut), ref) for cut in C.cuts_of(ref)]
        out.append(("pc_" + ref, ref))
    return out


def job(a):
    setname, floor = a
    for k in range(1, K + 1):
        if os.path.exists(os.path.join(ROOT, "results", setname, "A_k%d_%s.jsonl" % (k, floor))):
            continue
        seq, seq_dir, T_trues, trp = pl.prep(setname, "A", k)
        pl.register(setname, "A", k, seq, seq_dir, T_trues, trp, floor)
        # 2026-10-14 追記（実行の途中で C: が埋まり WSL が止まったため）：前処理の入力の点群と平面分割の出力（`<seq>/`）は
        # 登録の後に使わないので消す（R46 段階3 と同じ）。採点に使う T_rp（`processed/<seq>/30m/T_rp.json`）と結果は残す
        shutil.rmtree(os.path.join(ROOT, seq), ignore_errors=True)
    return a


def main():
    npar = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    jobs = sets(); first = {}
    for a in jobs:
        first.setdefault(a[1], a)
    rest = [a for a in jobs if a not in first.values()]
    with mp.get_context("spawn").Pool(npar) as p:
        for a in p.imap_unordered(job, list(first.values())):     # 参照ごとに 1 組（参照どうしはぶつからない）
            print("warm", a, flush=True)
        for i, a in enumerate(p.imap_unordered(job, rest)):
            print("done", i + 1, "/", len(rest), a, flush=True)
    print("all done")


if __name__ == "__main__":
    main()
