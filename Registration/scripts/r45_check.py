"""R45 §3 の確認：centroid（translation_init 未指定）の行列が R27 の proposed とビット一致するか、摂動 P が R27 と一致するか。

引数：出力の根（`output/Registration/r45_pre` または `output/Registration/r45`）。案A 側の P も R27 と比べる。
時間：各試行の time_s（trials.csv）の中央・最大、300 s を超えた件数。
"""
import csv, json, os, sys
import numpy as np

os.chdir(os.path.expanduser("~/rizu/SNI-SLAM"))
SCENES = ["room_0", "room_1", "room_2", "office_0", "office_1", "office_2", "office_3", "office_4"]
root = sys.argv[1]
res = {"root": root, "scenes": {}}
for s in SCENES:
    r27 = {t["trial"]: t for t in json.load(open("output/Registration/gt_a/%s/trial_matrices.json" % s))["trials"]
           if t["method"] == "proposed"}
    ent = {}
    for mode in ("centroid", "plan_a"):
        d = "%s/%s/%s" % (root, mode, s)
        tr = [t for t in json.load(open(d + "/trial_matrices.json"))["trials"] if t["method"] == "proposed"]
        p_eq = sum(np.array_equal(np.asarray(t["P"]), np.asarray(r27[t["trial"]]["P"])) for t in tr if t["trial"] >= 0)
        n = sum(1 for t in tr if t["trial"] >= 0)
        e = {"n": n, "P_equal": int(p_eq)}
        if mode == "centroid":
            e["T_bit_equal"] = int(sum(np.array_equal(np.asarray(t["T_est"]), np.asarray(r27[t["trial"]]["T_est"])) for t in tr))
            e["n_incl_T0"] = len(tr)
        times = [float(x["time_s"]) for x in csv.DictReader(open(d + "/trials.csv")) if x["method"] == "proposed"]
        e["time_median_s"] = float(np.median(times)); e["time_max_s"] = float(np.max(times)); e["n_over_300s"] = int(sum(t > 300 for t in times))
        ent[mode] = e
    res["scenes"][s] = ent
    print(s, ent)
json.dump(res, open("Registration/output/diag/%s_check.json" % os.path.basename(root), "w"), indent=1)
