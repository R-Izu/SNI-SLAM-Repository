"""GT-A の 8 シーンの来歴と残差を1枚の表にする。"""

import json
import os

os.chdir("/home/student/rizu/SNI-SLAM")
SCENES = ["room_0", "room_1", "room_2", "office_0", "office_1",
          "office_2", "office_3", "office_4"]

print("%-10s %8s %9s %11s %11s %11s %9s"
      % ("シーン", "点数", "フレーム", "自己検査[m]", "全点[m]",
         "オラクル[m]", "内包率"))
print("-" * 76)
rows = []
for s in SCENES:
    p = "output/GT_A/%s_manifest.json" % s
    if not os.path.exists(p):
        print("%-10s **無し**" % s)
        continue
    m = json.load(open(p))
    rows.append(m)
    print("%-10s %8d %9d %11.4f %11.4f %11.4f %9.4f"
          % (s, m["n_points"], m["n_frames_used"],
             m["selfcheck_chamfer_no_transform_m"],
             m["selfcheck_chamfer_all_points_m"],
             m["oracle_chamfer_keep_classes_m"],
             m["oracle_class_inlier_ratio"]))

if rows:
    o = [m["oracle_chamfer_keep_classes_m"] for m in rows]
    print("\nオラクル残差（参照と同じクラス）: 範囲 [%.4f, %.4f] m / 中央 %.4f m"
          % (min(o), max(o), sorted(o)[len(o) // 2]))
    print("**平均に潰さない。シーン別に上に出している**")
    same = all(m["M_applied"] == rows[0]["M_applied"] for m in rows)
    print("\n8 シーンに同じ既知 Sim(3) を掛けているか: %s" % ("はい" if same else "**いいえ**"))
    print("sim3_seed = %s" % rows[0]["sim3_seed"])
    print("提案手法を使っているか: %s"
          % ("いいえ" if not any(m["uses_proposed_method"] for m in rows) else "**はい**"))
