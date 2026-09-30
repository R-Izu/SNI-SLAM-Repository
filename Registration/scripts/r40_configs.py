"""R40 の config を作る（R39 §5 の config の写し。差し替えるのは参照・source・試行数・出力先だけ）。

§3 E2'：参照を 410 だけ（`spaces: ['410']`、キャッシュ `m3_ifc_410.npz`）。既定・案A × 100 試行
§4    ：参照 411（E2 相当）、案A の config。source を `output/R40_BIM/<入力>.ply` に替え、50 試行
"""
import copy, os, yaml
os.chdir(os.path.expanduser("~/rizu/SNI-SLAM"))
OUT = "Registration/configs/r40_bim"
os.makedirs(OUT, exist_ok=True)
INPUTS = ["zero", "N_1cm", "N_2cm", "N_5cm", "L_5", "L_10", "L_20", "Obg_10", "Obg_25", "Obg_50",
          "Owall_10", "Owall_25", "Owall_50", "S_0p5", "S_1p0", "S_2p0"]


def dump(c, name, note):
    with open(os.path.join(OUT, name + ".yaml"), "w") as f:
        f.write("# R40（自動生成。手で編集しない）。%s\n" % note)
        yaml.safe_dump(c, f, allow_unicode=True, sort_keys=False)
    print("wrote", name)


for mode in ("centroid", "plan_correlate"):
    c = copy.deepcopy(yaml.safe_load(open("Registration/configs/r39_bim/bim_E2_%s.yaml" % mode)))
    c["reference"] = dict(c["reference"], spaces=["410"], cache_path="Registration/output/ifc/m3_ifc_410.npz")
    c["eval"] = dict(c["eval"], out_dir="output/Registration/r40_bim/E2p_%s" % mode)
    dump(c, "E2p_%s" % mode, "r39_bim/bim_E2_%s.yaml の参照だけを 410 に替えた" % mode)

base = yaml.safe_load(open("Registration/configs/r39_bim/bim_E2_plan_correlate.yaml"))
for name in INPUTS:
    c = copy.deepcopy(base)
    c["source"] = {"type": "points_ply", "mesh_path": "output/R40_BIM/%s.ply" % name}
    c["eval"] = dict(c["eval"], trials=50, out_dir="output/Registration/r40_bim/err_%s" % name)
    dump(c, "err_%s" % name, "r39_bim/bim_E2_plan_correlate.yaml の source と試行数（50）だけを替えた")
