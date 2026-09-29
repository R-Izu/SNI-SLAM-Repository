"""R39 §5-3 — BIM 由来の source 用の config を作る。

実データの config（`realdata/m3_cor_a__E2.yaml`＝参照 411、`__E3.yaml`＝参照 410＋411）の写しで、
**source と正解の場所・出力先だけ**差し替える。手法のパラメータ・摂動の範囲・成功閾値・試行数・seed は実データと同一
（摂動は GT-A と同じ生成器・同じ範囲：回転 0〜180°・並進 ±2 m・log 縮尺 ±0.4）。

    python Registration/scripts/r39_bim_configs.py
"""
import copy, os, yaml

os.chdir(os.path.expanduser("~/rizu/SNI-SLAM"))
OUT = "Registration/configs/r39_bim"
os.makedirs(OUT, exist_ok=True)
for ref, tmpl in (("E2", "m3_cor_a__E2"), ("E3", "m3_cor_a__E3")):
    base = yaml.safe_load(open("Registration/configs/realdata/%s.yaml" % tmpl))
    for mode in ("centroid", "plan_correlate"):
        c = copy.deepcopy(base)
        c["source"] = {"type": "points_ply", "mesh_path": "output/R39_BIM/bim_source.ply"}
        c.setdefault("proposed", {})["translation_init"] = mode
        c["eval"] = dict(c["eval"], t_gt_path="output/R39_BIM/T_gt_bim.json",
                         out_dir="output/Registration/r39_bim/%s_%s" % (ref, mode))
        c.pop("diagnostics", None)
        path = os.path.join(OUT, "bim_%s_%s.yaml" % (ref, mode))
        with open(path, "w") as f:
            f.write("# R39 §5 段階1（自動生成。手で編集しない）。realdata/%s.yaml の写しで、\n"
                    "# source・eval.t_gt_path・eval.out_dir・proposed.translation_init だけ差し替えた。\n" % tmpl)
            yaml.safe_dump(c, f, allow_unicode=True, sort_keys=False)
        print("wrote", path, "| ref spaces", c["reference"].get("spaces"), "| eval", {k: c["eval"][k] for k in ("trials", "seed", "perturb")})
