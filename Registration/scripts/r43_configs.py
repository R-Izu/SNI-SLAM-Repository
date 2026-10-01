"""R43 の対照の config（R40 §4 の config の写し。差し替えるのは source・label_filter・出力先だけ）。

入力：誤差なし（`output/R40_BIM/zero.ply`、R40 §4 と同じ）と、`output/R43_BIM/F_{10,25,50}.ply`（411 の壁沿いの家具）
手法：`proposed`（案A＝`translation_init: plan_correlate`）を label_filter on と off で。50 試行（R40 と同じ生成器・seed 0 の先頭 50 個）
"""
import copy, os, yaml
os.chdir(os.path.expanduser("~/rizu/SNI-SLAM"))
OUT = "Registration/configs/r43_bim"
os.makedirs(OUT, exist_ok=True)
INPUTS = {"F_0": "output/R40_BIM/zero.ply", "F_10": "output/R43_BIM/F_10.ply",
          "F_25": "output/R43_BIM/F_25.ply", "F_50": "output/R43_BIM/F_50.ply"}

base = yaml.safe_load(open("Registration/configs/r40_bim/err_zero.yaml"))
assert base["proposed"]["translation_init"] == "plan_correlate" and base["eval"]["trials"] == 50
for name, ply in INPUTS.items():
    for lf in ("on", "off"):
        c = copy.deepcopy(base)
        c["source"] = {"type": "points_ply", "mesh_path": ply}
        c["proposed"] = dict(c["proposed"], label_filter={"enabled": lf == "on"})
        c["eval"] = dict(c["eval"], out_dir="output/Registration/r43_bim/%s_%s" % (name, lf))
        with open(os.path.join(OUT, "%s_%s.yaml" % (name, lf)), "w") as f:
            f.write("# R43（自動生成。手で編集しない）。r40_bim/err_zero.yaml の source・label_filter・出力先だけを替えた\n")
            yaml.safe_dump(c, f, allow_unicode=True, sort_keys=False)
        print("wrote", name, lf)
