"""R55 段階0：作り直した STSL の登録（`r46_reg_dump`）が R46 の陽性対照の結果を再現するか（libim の環境）。**回す前に commit する。**

- 平面分割（`r46_plane_seg`）は seed を固定できない（R46 §8-5、`r46_stsl_pipeline.py` の docstring）。そのため 1e-6 の比較は、
  **R46 で保存した前処理の済んだ入力**（`$ROOT/processed/pc_<S>_k<k>/`・`$ROOT/pc_<S>_k<k>/`）と登録の config
  （STSL の `configs/interval/r46/pc_<S>_k<k>_M3_410_411.yaml`）に、作り直した `r46_reg_dump` だけを回して行う
- 比べるもの：R46 の `$ROOT/results/pc/<S>_k<k>_M3_410_411.jsonl` の各行の `tf`（4×4）・`success_stsl`・`optimal_score`・`n_corres`・`n_corners`・
  候補の検証スコア。変換は要素ごとの差の最大
- 出力：`$ROOT/results/pc_r55_stage0/`（R46 の結果は上書きしない）、まとめ `Registration/output/diag/r55/stage0.json`
"""
import json, os, subprocess, sys
import numpy as np

ROOT = os.path.expanduser("~/rizu/r46_stsl_data")
STSL = os.path.expanduser("~/rizu/LiDAR2BIM-Registration")
BUILD = os.path.expanduser("~/rizu/r46_stsl_build")
OUT = os.path.join(ROOT, "results", "pc_r55_stage0")
DIAG = os.path.expanduser("~/rizu/SNI-SLAM/Registration/output/diag/r55")


def load(p):
    return [json.loads(l.replace("-inf", "-Infinity").replace("inf", "Infinity")) for l in open(p) if l.strip()]


def main():
    os.environ["LiBIM_UST_ROOT"] = ROOT.rstrip("/") + "/"
    os.makedirs(OUT, exist_ok=True); os.makedirs(DIAG, exist_ok=True)
    rows, worst = [], 0.0
    for S in ("A", "B"):
        for k in range(1, 6):
            rc = os.path.join(STSL, "configs", "interval", "r46", "pc_%s_k%d_M3_410_411.yaml" % (S, k))
            out = os.path.join(OUT, "%s_k%d_M3_410_411.jsonl" % (S, k))
            r = subprocess.run([os.path.join(BUILD, "r46_reg_dump"), rc, out], capture_output=True, text=True, cwd=STSL,
                               env=dict(os.environ))
            if r.returncode != 0:
                rows.append({"S": S, "k": k, "error": (r.stdout[-1500:] + r.stderr[-1500:])}); continue
            old = load(os.path.join(ROOT, "results", "pc", "%s_k%d_M3_410_411.jsonl" % (S, k))); new = load(out)
            for a, b in zip(old, new):
                d = float(np.max(np.abs(np.asarray(a["tf"], float) - np.asarray(b["tf"], float))))
                worst = max(worst, d)
                same = {f: a.get(f) == b.get(f) for f in ("success_stsl", "optimal_score", "n_corres", "n_corners")}
                ca = [c["score"] for c in a.get("candidates", [])]; cb = [c["score"] for c in b.get("candidates", [])]
                rows.append({"S": S, "k": k, "index": a["index"], "tf_max_abs_diff": d, **{"same_" + f: v for f, v in same.items()},
                             "cand_same": ca == cb, "n_cand": len(ca)})
            print(S, k, "n", len(new), "vs", len(old), flush=True)
    summ = {"n_rows": len(rows), "errors": [r for r in rows if "error" in r], "tf_max_abs_diff": worst,
            "all_fields_same": all(all(v for kk, v in r.items() if kk.startswith("same_") or kk == "cand_same") for r in rows if "error" not in r),
            "rows": rows}
    json.dump(summ, open(os.path.join(DIAG, "stage0.json"), "w"), indent=1, default=float)
    print("rows", len(rows), "errors", len(summ["errors"]), "tf max diff", worst, "all fields same", summ["all_fields_same"])


if __name__ == "__main__":
    main()
