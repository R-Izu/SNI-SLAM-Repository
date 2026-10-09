"""R45：GT-A を案A（translation_init: plan_correlate）と centroid で回す config。

`Registration/configs/gt_a/<scene>.yaml`（R27）の写し。R45 §6（main の回答 (b)）により、両方式とも
`reference.seed: 0` を足す（R30 §4 の `room_0_seeded.yaml` と同じ作法。参照の標本化を固定する）。
- `gt_a_centroid_seeded/<scene>.yaml`：`reference.seed: 0` だけを足す
- `gt_a_plan_a/<scene>.yaml`：`reference.seed: 0` と `proposed.translation_init: plan_correlate` を足す
それ以外（source・摂動・試行数・成功閾値）は R27 と同じ。
"""
import copy, os, yaml
os.chdir(os.path.expanduser("~/rizu/SNI-SLAM"))
SCENES = ["room_0", "room_1", "room_2", "office_0", "office_1", "office_2", "office_3", "office_4"]
for d in ("gt_a_plan_a", "gt_a_centroid_seeded"):
    os.makedirs("Registration/configs/%s" % d, exist_ok=True)
for s in SCENES:
    base = yaml.safe_load(open("Registration/configs/gt_a/%s.yaml" % s))
    assert "translation_init" not in (base.get("proposed") or {}) and "seed" not in base["reference"], s
    cen = copy.deepcopy(base); cen["reference"] = dict(cen["reference"], seed=0)
    pa = copy.deepcopy(cen); pa.setdefault("proposed", {})["translation_init"] = "plan_correlate"
    for d, c, note in (("gt_a_centroid_seeded", cen, "reference.seed: 0 だけを足した"),
                       ("gt_a_plan_a", pa, "reference.seed: 0 と proposed.translation_init: plan_correlate を足した")):
        with open("Registration/configs/%s/%s.yaml" % (d, s), "w") as f:
            f.write("# R45（自動生成。手で編集しない）。gt_a/%s.yaml に %s\n" % (s, note))
            yaml.safe_dump(c, f, allow_unicode=True, sort_keys=False)
    print("wrote", s)
