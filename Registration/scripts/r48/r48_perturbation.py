"""R48 §7：20 摂動の回転の角・軸の傾き・STSL に渡したヨー・縮尺・並進と、dev < 5° の内訳（事後、登録はしない）"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.expanduser("~/rizu/SNI-SLAM/Registration/scripts/r48"))
import r48_common as C

man = json.load(open(os.path.join(C.D, "inputs_manifest.json"))); Gm = np.asarray(man["G"])
gta = np.load(os.path.join(C.D, "gta.npz"))
Ps = C.perturbations()
rows = []
for k, P in enumerate(Ps):
    R, t, s = C.decompose_sim3(P)
    ang = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]); ax = w / (np.linalg.norm(w) + 1e-12)
    tilt_axis = float(np.degrees(np.arccos(min(1.0, abs(ax[2])))))
    tm = C.stsl_trial(P, Gm)
    # 水平化した P（Rg' P Rg'^-1）の、z まわりの twist 以外の部分＝傾き（STSL では捨てた分）
    Rl = C.decompose_sim3(tm["Rg"] @ P @ np.linalg.inv(tm["Rg"]))[0]
    tilt_lv = float(np.degrees(np.arccos(np.clip(Rl[2, 2], -1, 1))))
    phi = C.twist_z(C.decompose_sim3(tm["T_true"])[0])
    gpsi = {s_: round(C.stsl_trial(P, gta["G_" + s_])["psi"], 1) for s_ in C.GTA}
    rows.append({"k": k, "angle_deg": ang, "axis_tilt_from_z_deg": tilt_axis, "scale": s, "t": t.tolist(),
                 "psi_multi_deg": tm["psi"], "tilt_dropped_deg": tilt_lv, "Ttrue_yaw_multi_deg": phi, "psi_gta_deg": gpsi})
print("k | 角 | 軸と z | 縮尺 | 並進 | ψ(複数部屋) | 捨てた傾き | T_true の z 回転")
for r in rows:
    print("%2d | %6.1f | %5.1f | %.3f | %s | %7.1f | %6.1f | %7.1f" % (r["k"], r["angle_deg"], r["axis_tilt_from_z_deg"], r["scale"],
          np.round(r["t"], 2).tolist(), r["psi_multi_deg"], r["tilt_dropped_deg"], r["Ttrue_yaw_multi_deg"]))
print("psi GT-A", [r["psi_gta_deg"] for r in rows][:3])
# dev < 5° の内訳
sc = json.load(open(os.path.expanduser("~/rizu/SNI-SLAM/Registration/output/diag/r48/r48_score.json")))
by_k = np.zeros(20, int); n_k = np.zeros(20, int); theta = {}
for ref, e in sc["refs"].items():
    theta[ref] = man["refs"][ref]["theta0_deg"]
    for c, v in e["cells"].items():
        for k, d in enumerate(v["dev"]):
            n_k[k] += 1; by_k[k] += d < 5
print("dev<5 by k:", by_k.tolist(), "of", n_k.tolist(), "total", by_k.sum())
th = sorted(set(round(x, 1) for x in theta.values())); print("theta0 values", th)
# 複数部屋の dev 表（θ0 ごと）
for t0 in sorted(set(theta[r] for r in theta if not r.startswith("GTA_"))):
    devs = [C.dev_of(C.stsl_trial(P, Gm)["T_true"], t0) for P in Ps]
    print("theta0 %.1f dev" % t0, [round(x, 1) for x in devs], "n<5", sum(x < 5 for x in devs))
for s_ in C.GTA:
    t0 = theta["GTA_" + s_]; devs = [C.dev_of(C.stsl_trial(P, gta["G_" + s_])["T_true"], t0) for P in Ps]
    print("GTA %s theta0 %.1f dev" % (s_, t0), [round(x, 1) for x in devs], "n<5", sum(x < 5 for x in devs))
# T_true の z 回転 φ の mod 90 の分布
ph = np.array([r["Ttrue_yaw_multi_deg"] for r in rows]); print("phi mod 90", np.round(ph % 90, 1).tolist())
json.dump({"rows": rows, "dev_lt5_by_k": by_k.tolist(), "n_by_k": n_k.tolist()}, open(os.path.join(C.D, "posthoc", "perturbations.json"), "w"), indent=1)
