"""R58 段階1：R55 の STSL の登録だけを、候補の書き出しを足した `r58_reg_dump` で回し直す（libim の環境）。**回す前に commit する。**

- 前処理は回さない。R55 で保存した `$ROOT/processed/r55_<target>_f<f>_A_k<k>/`・`$ROOT/r55_<target>_f<f>_A_k<k>/` と、
  R55 の登録の config（STSL の `configs/interval/r46/r55_<target>_f<f>_A_k<k>_<floor>.yaml`）をそのまま使う
- 出力：`$ROOT/results/r58/r55_<target>_f<f>/A_k<k>_<floor>.jsonl`（R55 の結果は上書きしない）
使い方：python r58_run.py
"""
import os, subprocess

ROOT = os.path.expanduser("~/rizu/r46_stsl_data")
STSL = os.path.expanduser("~/rizu/LiDAR2BIM-Registration")
BIN = os.path.expanduser("~/rizu/r46_stsl_build/r58_reg_dump")
REAL = ["m3_cor_%s__%s" % (s, e) for s in "abcd" for e in ("E2", "E3")]
FLOOR = {"E2": "M3_411_both", "E3": "M3_410_411_both"}


def main():
    env = dict(os.environ, LiBIM_UST_ROOT=ROOT.rstrip("/") + "/")
    for t in REAL:
        fl = FLOOR[t.split("__")[1]]
        for f in ("1p0", "0p6"):
            setname = "r55_%s_f%s" % (t, f)
            out = os.path.join(ROOT, "results", "r58", setname); os.makedirs(out, exist_ok=True)
            for k in range(1, 6):
                rc = os.path.join(STSL, "configs", "interval", "r46", "%s_A_k%d_%s.yaml" % (setname, k, fl))
                o = os.path.join(out, "A_k%d_%s.jsonl" % (k, fl))
                r = subprocess.run([BIN, rc, o], capture_output=True, text=True, cwd=STSL, env=env)
                if r.returncode != 0:
                    raise RuntimeError(setname + " k%d\n" % k + r.stdout[-1500:] + r.stderr[-1500:])
            print(setname, "done", flush=True)


if __name__ == "__main__":
    main()
