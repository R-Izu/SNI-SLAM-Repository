"""R65 §3 の 1・2：位置合わせの出力データの一覧表（`01_Research/2026-11-20_registration_output_catalog.md`）の各行のファイルを開き、
欄（キー）・形・型を書き出す（sni-slam の環境）。**回す前に commit する。位置合わせは回さない。コードは変えない。**

- json：辞書のキー、リストの長さと先頭の要素の形を再帰で（数のリストは形 (n, m) と型）。辞書の値が同じ形の行（"0"〜"9" など）のときは、先頭の 1 つと数だけ
- jsonl：先頭の 1 行と行の数
- npz：キーごとの shape・dtype
- ply：ヘッダ
- 入力・参照・正解：`r49_run.config("real", "m3_cor_a__E2", "full")` などの source・reference の指定と、そのファイルの中身
出力：`~/rizu/r65_data/catalog.json`（ファイルごと）と、標準出力
"""
import glob, json, os, sys
import numpy as np

REPO = os.path.expanduser("~/rizu/SNI-SLAM")
for p in ("Registration", "Registration/scripts", "Registration/scripts/r49"):
    sys.path.insert(0, os.path.join(REPO, p))
os.chdir(REPO)
H = os.path.expanduser("~")
DIAG = "Registration/output/diag"


def shape(x, depth=0, maxd=7):
    if isinstance(x, dict):
        ks = list(x.keys())
        if len(ks) > 6 and all(k.lstrip("-").isdigit() for k in ks):
            return {"<dict of %d rows, keys %s..%s>" % (len(ks), ks[0], ks[-1]): shape(x[ks[0]], depth + 1, maxd)}
        if depth >= maxd:
            return "dict(%d keys)" % len(ks)
        return {k: shape(v, depth + 1, maxd) for k, v in list(x.items())[:40]}
    if isinstance(x, list):
        if not x:
            return "list(0)"
        try:
            a = np.asarray(x)
            if a.dtype.kind in "fiub" and a.ndim >= 1:
                return "array%s %s" % (tuple(a.shape), a.dtype)
        except Exception:
            pass
        return {"<list(%d)>" % len(x): shape(x[0], depth + 1, maxd)}
    if isinstance(x, float):
        return "float"
    if isinstance(x, bool):
        return "bool"
    if isinstance(x, int):
        return "int"
    if isinstance(x, str):
        return "str"
    return type(x).__name__


def describe(p):
    d = {"path": p.replace(H, "~"), "bytes": os.path.getsize(p)}
    if p.endswith(".json"):
        d["schema"] = shape(json.load(open(p)))
    elif p.endswith(".jsonl"):
        ls = open(p).read().splitlines()
        d["n_lines"] = len(ls)
        d["schema"] = shape(json.loads(ls[0].replace("-inf", "-Infinity").replace("inf", "Infinity"))) if ls else None
    elif p.endswith(".npz"):
        z = np.load(p, allow_pickle=False)
        d["arrays"] = {k: [list(z[k].shape), str(z[k].dtype)] for k in z.files[:60]}; d["n_arrays"] = len(z.files)
    elif p.endswith(".ply"):
        h = open(p, "rb").read(1500); d["header"] = h[:h.find(b"end_header")].decode("ascii", "replace")
    elif p.endswith(".csv"):
        d["head"] = open(p).read().splitlines()[:3]
    return d


def first(pattern):
    g = sorted(glob.glob(os.path.expanduser(pattern), recursive=True)); return g[0] if g else None


ROWS = {
    "3 本手法・完全 R49 実": "~/rizu/r49_data/results/real/full/*.json",
    "3 本手法・完全 R49 合成": "~/rizu/r49_data/results/synth/full/*.json",
    "3 R49 bim": "~/rizu/r49_data/results/bim/full/*.json",
    "3 R49 実 fpfh_p2l": "~/rizu/r49_data/results/real/fpfh_p2l/*.json",
    "3 R52": "~/rizu/r52_data/results/f0p6/*.json",
    "3 R50 実 score_only": "~/rizu/r50_data/results/real/score_only/*.json",
    "3 R50 tilt": "~/rizu/r50_data/tilt/real.json",
    "3 R55 prop_A": "~/rizu/r55_data/global/prop_A/*.json",
    "3 R55 fpfh_A": "~/rizu/r55_data/global/fpfh_A/*.json",
    "3 R55 trials": "~/rizu/r55_data/trials.json",
    "3 R56 trials": "~/rizu/r56_data/trials.json",
    "3 R56 cor_d": "~/rizu/r56_data/cor_d.json",
    "3 STSL R55": "~/rizu/r46_stsl_data/results/r55_*/A_k1_*.jsonl",
    "3 STSL R58": "~/rizu/r46_stsl_data/results/r58/r55_*/A_k1_*.jsonl",
    "3 R63 条件B fix": "~/rizu/r63_data/results_fix/full/*.json",
    "3 R63 条件A fix": "~/rizu/r63_data/resultsA_fix/full/*.json",
    "4 R54 trials": "~/rizu/r54_data/trials.json",
    "4 R60 trials": "~/rizu/r60_data/trials.json",
    "4 R60 summary": DIAG + "/r60/r60_summary.json",
    "4 R58 trials": "~/rizu/r58_data/trials.json",
    "4 R63 trialsA_fix": "~/rizu/r63_data/trialsA_fix.json",
    "4 R63 summaryA_fix": DIAG + "/r63/summaryA_fix.json",
    "2 omega": DIAG + "/omega.npz",
    "2 r39_omega_bim": DIAG + "/r39_omega_bim.npz",
    "2 R52 inputs": "~/rizu/**/r52_inputs.json",
    "その他 R51": "~/rizu/r51_data/stage0_trials.json",
    "その他 R53 main": "~/rizu/r53_data/main.json",
    "その他 R53 controls": "~/rizu/r53_data/controls.json",
}


def main():
    os.makedirs(os.path.expanduser("~/rizu/r65_data"), exist_ok=True)
    out = {"rows": {}, "inputs": {}}
    for name, pat in ROWS.items():
        pat = pat if pat.startswith("~") else os.path.join(REPO, pat)
        p = first(pat)
        n = len(glob.glob(os.path.expanduser(pat), recursive=True))
        out["rows"][name] = {"pattern": pat.replace(H, "~"), "n_files": n, "first": describe(p) if p else None}
    # 入力・参照・正解（config から）
    import r49_run as RR
    for kind, tgt in (("real", "m3_cor_a__E2"), ("real", "m3_cor_a__E3"), ("synth", None)):
        try:
            if tgt is None:
                tgts = RR.targets("synth") if hasattr(RR, "targets") else []
                tgt = tgts[0] if tgts else None
            cfg = RR.config(kind, tgt, "full")
            e = {"target": tgt, "source": cfg.get("source"), "reference": cfg.get("reference"), "gt": {k: v for k, v in cfg.items() if "gt" in k.lower()}}
            for key in ("source", "reference"):
                for f in ("mesh_path", "path", "cache_path", "traj_path"):
                    v = (cfg.get(key) or {}).get(f)
                    if isinstance(v, str) and os.path.exists(v) and os.path.isfile(v) and os.path.getsize(v) < 3e9:
                        e["%s.%s" % (key, f)] = describe(v)
            out["inputs"]["%s|%s" % (kind, tgt)] = e
        except Exception as ex:
            out["inputs"]["%s|%s" % (kind, tgt)] = {"error": repr(ex)}
    json.dump(out, open(os.path.expanduser("~/rizu/r65_data/catalog.json"), "w"), indent=1, ensure_ascii=False, default=str)
    print(json.dumps(out, indent=1, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
