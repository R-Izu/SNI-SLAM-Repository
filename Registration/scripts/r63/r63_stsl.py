"""R63：ConSLAM で STSL の登録を回す（libim の環境、ブランチ `exp/r63`）。**回す前に commit する。** 指示書 §3-1。

- 入力と前処理：R62 で作ったもの（`$ROOT/inputs/r62_b2_f<f>/`、`$ROOT/processed/r62_b2_f<f>_A_k<k>/`、k = 1〜5）。前処理は回し直さない
- BIM：R62 の `$ROOT/BIM/CS_<参照>/`（参照 4 つ）
- 登録：`r46_stsl_pipeline.register` をそのまま使い、呼ぶ実行ファイルだけを、候補の tf を書き出す `r58_reg_dump`（R58）にする
  （`pl.BUILD` を、`r46_reg_dump` という名前で `r58_reg_dump` を指す一時のフォルダに替える）。設定 A（R55 の条件 A と同じ）
- T_trues は入力の `spec.json` の `T_true`、T_rp は前処理の `T_rp.json`
出力：`$ROOT/results/r62_b2_f<f>/A_k<k>_CS_<参照>.jsonl`
"""
import json, os, sys, tempfile
import numpy as np

REPO = os.path.expanduser("~/rizu/SNI-SLAM")
ROOT = os.path.expanduser("~/rizu/r46_stsl_data")
os.environ["R46_STSL_ROOT"] = ROOT
sys.path.insert(0, os.path.join(REPO, "Registration", "scripts", "r46_stsl"))
import r46_stsl_pipeline as pl                                   # noqa: E402

REFS = ["S1_small", "S1_large", "S3_small", "S3_large"]


def main():
    tmp = tempfile.mkdtemp(prefix="r63_bin_")
    os.symlink(os.path.expanduser("~/rizu/r46_stsl_build/r58_reg_dump"), os.path.join(tmp, "r46_reg_dump"))
    pl.BUILD = tmp
    for f in ("1p0", "0p6"):
        name = "r62_b2_f" + f
        spec = json.load(open(os.path.join(ROOT, "inputs", name, "spec.json")))
        T_trues = [np.asarray(t["T_true"], dtype=np.float64) for t in spec["trials"]]
        for k in range(1, 6):
            seq = "%s_A_k%d" % (name, k); seq_dir = os.path.join(ROOT, "processed", seq, "30m")
            trp_d = json.load(open(os.path.join(seq_dir, "T_rp.json")))
            trp = [np.asarray(trp_d[str(i)] if str(i) in trp_d else trp_d[i], dtype=np.float64) for i in range(len(T_trues))]
            for ref in REFS:
                out = pl.register(name, "A", k, seq, seq_dir, T_trues, trp, "CS_" + ref)
                print("done", out, flush=True)


if __name__ == "__main__":
    main()
