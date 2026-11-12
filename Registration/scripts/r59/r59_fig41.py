"""R59 §3-B：ゼミ資料 4.1 の図の点群を書き出す（sni-slam の環境、ブランチ `exp/r59`）。**回す前に commit する。位置合わせは回さない。**

試行の選び方（§3-B-1、固定）：R54 の `~/rizu/r54_data/trials.json` で、method full・data real・f 1p0・ref E2、主の線の g・s・a・f がすべて真で
returned_at が無い試行のうち、スキャン m3_cor_a → d の順、同じスキャンでは k の小さい順で最初の 1 つ
点：参照は `load_reference_cloud(cfg)`（E2、R49 の完全と同じ config）。SLAM は `preprocess.prepare(transform_cloud(src, P), cfg)`
（本手法に渡した後の前処理と同じ。voxel 0.05 m）。2 は P をかけた入力のまま、3〜6 は記録の変換 T をその点にかけたもの（BIM の座標）
記録：R49 の `rows[k]`（`diag.gen`・`diag.short`・`diag.after_icp`・`diag.scores`・`T_est`）
- 水平の向きの候補の番号：gen・short は cand_id // 4（`plan_correlate.generate_candidates` は向きごとに 4 つずつ、向きの順に並べる）、
  after_icp は i（`proposed.py:238` の `sorted(best_per_yaw)` の順）
- 記録の中の順位：gen・short は向きの中の番号 + 1（水平相関の値の順、`plan_correlate.py` の per_yaw の並べ方）。after_icp は `scores` の降順
- 採点の値：after_icp は記録の `scores`。short の `score_short` は R49 の記録に無いので、`class_inlier_ratio(src_p, dst_p, T, 0.3)` で再計算した値を
  別の欄（score_short_recomputed）に書く。gen は無し
§4 S6：最終（T_est）の e_R・e_Ω を R54 の値と比べ、差が 1e-6 を越えれば書き出さずに止める
出力：`Registration/output/fig/r59_fig41/<1_gt|2_initial|3_gen16|4_icp16|5_prune4|6_final>/`
"""
import csv, json, os, struct, sys
import numpy as np

REPO = os.path.expanduser("~/rizu/SNI-SLAM")
for p in ("Registration", "Registration/scripts", "Registration/scripts/r49"):
    sys.path.insert(0, os.path.join(REPO, p))
os.chdir(REPO)
from regbim import io_utils, metrics, preprocess                # noqa: E402
from r41_rescore import per_trial, real_criteria                # noqa: E402
import r49_run as RR                                            # noqa: E402

OUT = "Registration/output/fig/r59_fig41"
BLUE, ORANGE = (0, 114, 178), (230, 159, 0)


def write_ply(path, pts, rgb):
    pts = np.asarray(pts, dtype=np.float32)
    with open(path, "wb") as f:
        f.write(("ply\nformat binary_little_endian 1.0\nelement vertex %d\nproperty float x\nproperty float y\nproperty float z\n"
                 "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n" % len(pts)).encode())
        rec = np.zeros(len(pts), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
        rec["x"], rec["y"], rec["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
        rec["r"], rec["g"], rec["b"] = rgb
        f.write(rec.tobytes())


def preview(path, ref, clouds, title):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 8))
    rng = np.random.default_rng(0)
    r = ref[rng.choice(len(ref), min(len(ref), 30000), replace=False)]
    ax.scatter(r[:, 0], r[:, 1], s=0.3, c=[np.array(BLUE) / 255])
    for c in clouds:
        x = c[rng.choice(len(c), min(len(c), 8000), replace=False)]
        ax.scatter(x[:, 0], x[:, 1], s=0.3, c=[np.array(ORANGE) / 255], alpha=0.5)
    ax.set_aspect("equal"); ax.set_title(title); fig.tight_layout(); fig.savefig(path, dpi=100); plt.close(fig)


def main():
    R = json.load(open(os.path.expanduser("~/rizu/r54_data/trials.json")))["rows"]
    cand = [r for r in R if r["method"] == "full" and r["data"] == "real" and r["f"] == "1p0" and r["ref"] == "E2"]
    ok_rows = [r for r in cand if all(r["main"][x] for x in "gsaf") and not r["main"]["returned_at"]]
    why = {"n_E2_f1p0": len(cand), "n_all_stages_ok_not_returned": len(ok_rows),
           "n_returned": sum(bool(r["main"]["returned_at"]) for r in cand),
           "n_some_stage_0": sum(not all(r["main"][x] for x in "gsaf") for r in cand)}
    if not ok_rows:
        print("no trial", why); return
    sel = sorted(ok_rows, key=lambda r: (r["scan"], r["k"]))[0]
    target, k = sel["target"], sel["k"]
    cfg = RR.config("real", target, "full")
    rec = json.load(open(RR.path("real", "full", target)))["rows"][str(k)]
    P = np.asarray(rec["P"], dtype=np.float64); d = rec["diag"]
    G = np.asarray(real_criteria(sel["scan"])[1]); om = np.load("Registration/output/diag/omega.npz")[sel["scan"]]
    omb = np.load("Registration/output/diag/r39_omega_bim.npz")[target]
    pf = per_trial(rec["T_est"], P, G, om, omb)
    s6 = {"eR_diff": abs(pf["eR"] - sel["eR"]), "eO_diff": abs(pf["eO"] - sel["eO"])}
    info = {"selected": {"target": target, "scan": sel["scan"], "ref": sel["ref"], "k": k}, "why": why, "final": {x: float(pf[x]) for x in ("eR", "eO", "eOB", "q")},
            "r54": {"eR": sel["eR"], "eO": sel["eO"], "eOB": sel["eOB"]}, "s6": s6}
    print(json.dumps(info))
    if max(s6.values()) > 1e-6:
        json.dump(info, open(os.path.join(OUT, "_stopped_s6.json"), "w"), indent=1); return
    src = io_utils.load_source_cloud(cfg); dst = io_utils.load_reference_cloud(cfg)
    sp = preprocess.prepare(metrics.transform_cloud(src, P), cfg)
    dp = preprocess.prepare(dst, cfg)
    ref = np.asarray(dst.points)
    os.makedirs(OUT, exist_ok=True)
    files = {}
    def put(folder, name, pts, rgb):
        os.makedirs(os.path.join(OUT, folder), exist_ok=True)
        write_ply(os.path.join(OUT, folder, name), pts, rgb); files["%s/%s" % (folder, name)] = int(len(pts))
    put("1_gt", "bim_reference.ply", ref, BLUE)
    put("2_initial", "slam_initial.ply", sp.points, ORANGE)
    preview(os.path.join(OUT, "2_initial", "preview_top.png"), ref, [sp.points], "2_initial (input, before registration)")
    preview(os.path.join(OUT, "1_gt", "preview_top.png"), ref, [], "1_gt (BIM reference E2)")
    top = int(np.argsort(-np.asarray(d["scores"], dtype=float), kind="stable")[0])
    order4 = list(np.argsort(-np.asarray(d["scores"], dtype=float), kind="stable"))
    stages = [("3_gen16", d["gen"]), ("4_icp16", d["short"]), ("5_prune4", d["after_icp"]), ("6_final", [rec["T_est"]])]
    for folder, Ts in stages:
        rows, clouds = [], []
        for i, T in enumerate(Ts):
            T = np.asarray(T, dtype=np.float64); pts = metrics.apply_sim3(T, sp.points); clouds.append(pts)
            name = "slam_final.ply" if folder == "6_final" else "cand_%02d.ply" % i
            put(folder, name, pts, ORANGE)
            e = per_trial(T, P, G, om, omb); _, _, s = metrics.decompose_sim3(T)
            row = {"cand": i, "file": name,
                   "yaw_index": (i // 4 if folder in ("3_gen16", "4_icp16") else (i if folder == "5_prune4" else top)),
                   "rank_in_record": (i % 4 + 1 if folder in ("3_gen16", "4_icp16") else (order4.index(i) + 1 if folder == "5_prune4" else 1)),
                   **{"T%d%d" % (a, b): float(T[a, b]) for a in range(4) for b in range(4)}, "scale": float(s),
                   "eR_deg": float(e["eR"]), "eO_m": float(e["eO"]), "eOB_m": float(e["eOB"]),
                   "correct_main": int(bool(e["v"] and e["eR"] < 45 and e["eO"] < 1.0))}
            if folder == "4_icp16":
                row["score_short_recomputed"] = float(metrics.class_inlier_ratio(sp, dp, T, 0.3))
            if folder == "5_prune4":
                row["score_record"] = float(d["scores"][i]); row["selected"] = int(i == top)
            if folder == "6_final":
                row["score_record"] = float(d["scores"][top]); row["selected_after_icp_index"] = top
                row["T_est_eq_after_icp_top"] = int(np.array_equal(T, np.asarray(d["after_icp"][top])))
            rows.append(row)
        with open(os.path.join(OUT, folder, "transforms.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        preview(os.path.join(OUT, folder, "preview_top.png"), ref, clouds, "%s (%d candidates)" % (folder, len(Ts)))
    sizes = {}
    for folder in sorted(os.listdir(OUT)):
        p = os.path.join(OUT, folder)
        if os.path.isdir(p):
            sizes[folder] = sum(os.path.getsize(os.path.join(p, x)) for x in os.listdir(p))
    info.update(files=files, folder_bytes=sizes, total_bytes=sum(sizes.values()), n_ref_points=int(len(ref)), n_slam_points=int(len(sp.points)))
    json.dump(info, open(os.path.join(OUT, "_info.json"), "w"), indent=1)
    print("total MB", sum(sizes.values()) / 1e6, sizes)


if __name__ == "__main__":
    main()
