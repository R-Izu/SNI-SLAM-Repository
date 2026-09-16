import json, os, glob
for d in ["stages_m3_block_b__E2","stages_m3_cor_c__E2","stages_m3_cor_c__E2__vertical_only",
          "stages_m3_cor_d__E3__median_axes","stages_m3_cor_d__E3__vertical_only"]:
    p = "/d/rizu/SNI-SLAM/output/%s/STAGES.json" % d
    if not os.path.exists(p): continue
    m = json.load(open(p))
    print("=== %s ===" % d)
    print("  候補スコア %s  勝者 %s" % (m.get("candidate_scores"), m.get("winner")))
    for f in m["files"]:
        print("   %-34s 回転 %8s  縮尺比 %8s  d_Ω %8s"
              % (f["file"], f.get("gt_rot_deg","—"), f.get("scale_ratio","—"),
                 f.get("d_omega_m","—")))
