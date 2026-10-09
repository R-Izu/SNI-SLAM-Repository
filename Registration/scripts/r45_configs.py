"""R45：GT-A を案A（translation_init: plan_correlate）で回す config。

`Registration/configs/gt_a/<scene>.yaml`（R27）の写しで、`proposed.translation_init: plan_correlate` だけを足す。
それ以外（source・参照・摂動・試行数・成功閾値）は R27 と同じ。centroid 側は R27 の config をそのまま使う。
"""
import copy, os, yaml
os.chdir(os.path.expanduser("~/rizu/SNI-SLAM"))
SCENES = ["room_0", "room_1", "room_2", "office_0", "office_1", "office_2", "office_3", "office_4"]
OUT = "Registration/configs/gt_a_plan_a"
os.makedirs(OUT, exist_ok=True)
for s in SCENES:
    c = yaml.safe_load(open("Registration/configs/gt_a/%s.yaml" % s))
    assert "translation_init" not in (c.get("proposed") or {}), s
    c = copy.deepcopy(c)
    c.setdefault("proposed", {})["translation_init"] = "plan_correlate"
    with open(os.path.join(OUT, s + ".yaml"), "w") as f:
        f.write("# R45（自動生成。手で編集しない）。gt_a/%s.yaml に proposed.translation_init: plan_correlate だけを足した\n" % s)
        yaml.safe_dump(c, f, allow_unicode=True, sort_keys=False)
    print("wrote", s)
