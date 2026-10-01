"""R39 §4 の集計：`r39_criteria.json`（G1〜G8）から、表・壁との離れ（G8）・R37 の 3 区分（6 基準／8 基準）・PNG を出す。

- 壁との離れ：R38 §2-6 と同じ手順（`r38_map_bend.py` の該当部分と同じ規則）を、G8 で置いた点で行う
- R37：各試行の T（摂動を戻したもの）を G1〜G8 で採点。3 区分を「6 基準」と「8 基準」で数える
  （**8 基準の 3 区分は、結果を見た後に足した基準によるもの**）
"""
import copy, json, os, sys
import numpy as np, yaml
from scipy.spatial import cKDTree
os.chdir(os.path.expanduser("~/rizu/SNI-SLAM")); sys.path.insert(0, "Registration"); sys.path.insert(0, "Registration/scripts")
from criterion_verdict import d_omega
from failure_decomposition import provenance
from regbim import io_utils, metrics, preprocess
from regbim.labels import NAME_TO_ID

TH = {"rot_deg": 5.0, "trans": 0.1, "scale_ratio": 0.05}
C = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "Registration/output/diag/r39_criteria.json"))
S = C["scenes"]


def wall_offsets(scene, T):
    cfg = copy.deepcopy(yaml.safe_load(open("Registration/configs/realdata/%s__E2.yaml" % scene)))
    cfg["source"] = dict(cfg["source"], seed=0)
    src = preprocess.prepare(io_utils.load_source_cloud(cfg), cfg)
    dst = io_utils.load_reference_cloud(cfg)
    wall = src.subset(src.class_mask("wall"))
    R = metrics.decompose_sim3(T)[0]
    P = metrics.apply_sim3(T, wall.points)
    bnn = preprocess.prepare(dst, cfg)
    bw = bnn.points[bnn.labels == NAME_TO_ID["wall"]]; bn = bnn.normals[bnn.labels == NAME_TO_ID["wall"]]
    yw = bw[np.abs(bn[:, 1]) > 0.9]
    zlo = float(np.median(dst.points[dst.labels == NAME_TO_ID["floor"], 2]))
    zhi = float(np.median(dst.points[dst.labels == NAME_TO_ID["ceiling"], 2]))
    out = {}
    for name, yv, outward in (("south_far_from_corridor", yw[:, 1].min(), -1.0),
                              ("north_corridor_side", yw[:, 1].max(), +1.0)):
        sel = yw[np.abs(yw[:, 1] - yv) < 0.2]
        y0 = float(np.median(sel[:, 1])); x0, x1 = float(sel[:, 0].min()), float(sel[:, 0].max())
        ny = np.abs((wall.normals @ R.T)[:, 1])
        m = (ny > 0.8) & (np.abs(P[:, 1] - y0) < 0.5) & (P[:, 0] >= x0) & (P[:, 0] <= x1)
        bins = []
        for lo in np.arange(np.floor(x0), x1, 1.0):
            mm = m & (P[:, 0] >= lo) & (P[:, 0] < lo + 1.0)
            if mm.sum() < 20:
                bins.append({"x_lo": float(lo), "n": int(mm.sum()), "offset_m": None, "height_cov": None}); continue
            zs = P[mm, 2]; lv = np.linspace(zlo, zhi, 11)
            bins.append({"x_lo": float(lo), "n": int(mm.sum()),
                         "offset_m": float(np.median((P[mm, 1] - y0) * outward)),
                         "height_cov": float(np.mean([np.any((zs >= lv[k]) & (zs < lv[k + 1])) for k in range(10)]))})
        out[name] = {"y_m": y0, "x_range": [x0, x1], "bins": bins}
    return out


def score(T, Gs, om):
    R, _, s = metrics.decompose_sim3(T); degen = bool(metrics.is_degenerate_sim3(T))
    ok = {}
    for j, G in Gs.items():
        RG, _, sG = metrics.decompose_sim3(G)
        ok[j] = bool((not degen) and metrics.rotation_error_deg(R, RG) < TH["rot_deg"]
                     and abs(s / sG - 1) < TH["scale_ratio"] and d_omega(T, G, om) < TH["trans"])
    return ok


def three(ok, ids):
    n = sum(ok[j] for j in ids)
    return "all_pass" if n == len(ids) else ("all_fail" if n == 0 else "criterion_dependent")


def main():
    res = {"provenance": provenance(), "source_json_provenance": C["provenance"], "wall_offsets_g8": {}, "r37": {}}
    # 表
    print("## 回転の差（G1 基準、ヨー / 傾き [°]）と幅")
    for sc in sorted(S):
        r = S[sc]["rot_vs_g1"]
        print("%-11s " % sc + " ".join("G%s %+.2f/%.2f" % (j, r[j]["yaw_deg"], r[j]["tilt_deg"]) for j in sorted(r, key=int))
              + " | ヨー幅 6:%.2f 8:%.2f | G8 縮尺 %.4f" % (S[sc]["yaw_range_6"][1] - S[sc]["yaw_range_6"][0],
                                                     S[sc]["yaw_range_8"][1] - S[sc]["yaw_range_8"][0], S[sc]["g8"]["scale"]))
    print("\n## 基準対の dΩ / dΩ_BIM（G1 との対、G7-G8）")
    for sc in sorted(S):
        p = S[sc]["pairs"]
        print("%-11s " % sc + " ".join("1-%d %.3f/%.3f" % (j, p["1-%d" % j]["d"], p["1-%d" % j]["d_bim"]) for j in range(2, 9) if "1-%d" % j in p)
              + " | 7-8 %.3f/%.3f" % (p["7-8"]["d"], p["7-8"]["d_bim"]))
    # 壁との離れ（G8）
    for sc in ("m3_cor_b", "m3_cor_c", "m3_cor_d"):
        w = wall_offsets(sc, np.asarray(S[sc]["T"]["8"]))
        res["wall_offsets_g8"][sc] = w
        for k, v in w.items():
            print("  G8 %s %s: " % (sc, k) + " ".join("%.0f:%s(%s)" % (b["x_lo"], "—" if b["offset_m"] is None else "%+.3f" % b["offset_m"],
                                                        "—" if b["height_cov"] is None else "%.1f" % b["height_cov"]) for b in v["bins"]))
    # R37 の 3 区分
    z = np.load("Registration/output/diag/omega.npz")
    rows = json.load(open("Registration/output/diag/r37_replicate.json"))["rows"]
    from collections import defaultdict
    cnt = defaultdict(lambda: defaultdict(int))
    for r in rows:
        sc = r["target"].split("__")[0]
        Gs = {int(j): np.asarray(T) for j, T in S[sc]["T"].items()}
        P = np.asarray(r["P"]) if r["P"] else np.eye(4)
        for side in ("off", "on"):
            ok = score(np.asarray(r["T_" + side]) @ P, Gs, z[sc])
            cnt[(r["target"], side)]["6:" + three(ok, range(1, 7))] += 1
            cnt[(r["target"], side)]["8:" + three(ok, range(1, 9))] += 1
            cnt[(r["target"], side)]["G7_ok"] += ok[7]; cnt[(r["target"], side)]["G8_ok"] += ok[8]
    print("\n## R37 160 対の 3 区分（6 基準 / 8 基準）")
    for k in sorted(cnt):
        print(k, dict(cnt[k]))
        res["r37"]["%s|%s" % k] = dict(cnt[k])
    json.dump(res, open("Registration/output/diag/r39_criteria_report.json", "w"), indent=1, ensure_ascii=False, default=float)
    # PNG
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    sc_list = sorted(S)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=110)
    for i, sc in enumerate(sc_list):
        r = S[sc]["rot_vs_g1"]
        for j in sorted(r, key=int):
            jj = int(j); y = len(sc_list) - 1 - i
            col, mk = ("#2a78d6", "o") if jj <= 6 else (("#eb6834", "s") if jj == 7 else ("#1baf7a", "D"))
            ax.scatter(r[j]["yaw_deg"], y, c=col, marker=mk, s=28, edgecolors="none", zorder=3)
    ax.set_yticks(range(len(sc_list))); ax.set_yticklabels(list(reversed(sc_list)))
    ax.axvline(0, color="#898781", linewidth=0.8)
    ax.set_xlabel("yaw relative to G1 [deg]"); ax.grid(axis="x", color="#e1e0d9")
    from matplotlib.lines import Line2D
    ax.legend([Line2D([], [], marker="o", color="#2a78d6", linestyle=""), Line2D([], [], marker="s", color="#eb6834", linestyle=""),
               Line2D([], [], marker="D", color="#1baf7a", linestyle="")],
              ["G1–G6 (point-distance ICP, R10)", "G7 (wall-only ICP)", "G8 (wall-normal yaw)"], fontsize=8, frameon=False, loc="lower right")
    ax.set_title("Yaw of criteria G1–G8 relative to G1 (10 real-data scenes)")
    p = "/mnt/d/rizu/Obsidian-vault/MasterThesis_of_BIM_Vault/08_images/2026-09-30_r39_criteria_yaw.png"
    fig.savefig(p, bbox_inches="tight"); print("wrote", p, os.path.getsize(p))


if __name__ == "__main__":
    main()
