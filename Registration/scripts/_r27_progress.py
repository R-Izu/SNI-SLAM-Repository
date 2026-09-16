"""GT-A ベンチの途中経過。**終わったシーンだけを、母数つきで出す。**"""
import csv, os
os.chdir("/home/student/rizu/SNI-SLAM")
B = "output/Registration/gt_a"
M = ["proposed", "proposed_no_semantic", "proposed_no_gravity", "proposed_fixed_scale",
     "baseline_fgr_p2l", "baseline_fgr", "baseline_ransac_p2l", "baseline_open3d"]
scenes = [s for s in ["room_0","room_1","room_2","office_0","office_1","office_2",
                      "office_3","office_4"]
          if os.path.exists(os.path.join(B, s, "results.csv"))]
print("終わったシーン: %s（%d / 8）\n" % (", ".join(scenes), len(scenes)))
print("%-22s %s" % ("手法", "  ".join("%-9s" % s for s in scenes)))
print("-" * (23 + 11 * len(scenes)))
tot = {}
for m in M:
    cells, k, n = [], 0, 0
    for s in scenes:
        r = {x["method"]: x for x in csv.DictReader(open(os.path.join(B, s, "results.csv")))}
        if m in r and r[m].get("gt_success_rate", "") not in ("", None):
            v = float(r[m]["gt_success_rate"]); t = int(float(r[m]["robust_trials"]))
            cells.append("%-9.2f" % v); k += v * t; n += t
        else:
            cells.append("%-9s" % "—")
    tot[m] = (k, n)
    print("%-22s %s" % (m, "  ".join(cells)))
print("\n**GT 基準の成功率（途中集計。母数つき）**")
for m in M:
    k, n = tot[m]
    if n:
        print("  %-22s %5.1f / %d = **%.1f%%**" % (m, k, n, 100.0 * k / n))
