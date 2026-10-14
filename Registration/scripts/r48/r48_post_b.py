"""R48 §6 事後の確認 b（sni-slam の環境）：両方式が成功した試行で、推定と正解の差を水平の並進・鉛直の並進・ヨー・傾きに分ける。**事後。登録は回さない。**

分け方（すべての集合で同じ）
- T = 推定（source → 参照）、G = 正解。案A は T_est · P、STSL は Shift(+z_floor) · tf · T_rp · A · Rg（R48）／ A · Rg（R46・R47 B-1）／
  入力 → BIM の推定 · P_src_to_input（R47 B-2）
- 傾き・ヨー：`r36_rotation_diag.tilt_yaw(R_T, R_G)`（傾き＝上向きのずれ、ヨー＝z まわりの twist の絶対値）
- 並進：Ω（採点と同じ点）の重心 c を、T と G で参照の座標に置いたときの差 d = T c − G c。水平 = |d_xy|、鉛直 = |d_z|
- e_Ω（形 C の d_Ω）も並べる
- STSL の回の代表：K 回のうち多数決の側の回の中で e_Ω が中央値の回（R48 段階1 の対の差と同じ）
- 「成功」は R46・R47・R48 と同じ大きな失敗の基準（G1）の逆。実データの STSL は両面・設定 A（R47 の主）
集合
- R48：段階1 の 1,540 試行のうち両方式が成功したもの（1,001）
- 実データ R46・R47 B-1：R47 B-1 の 40 条件（うち 24 は R46 の再利用）＋ R46 の r = 8 m の 8 条件（同じ床の組）、各 10 試行
- 実データ R47 B-2：288 向き
出力：`Registration/output/diag/r48/r48_posthoc_b.json`
"""
import json, os, re, sys
from math import isfinite
import numpy as np

REPO = os.path.expanduser("~/rizu/SNI-SLAM")
for p in ("Registration", "Registration/scripts", "Registration/scripts/r47", "Registration/scripts/r48"):
    sys.path.insert(0, os.path.join(REPO, p))
os.chdir(REPO)
from r41_rescore import per_trial, real_criteria                # noqa: E402
from r36_rotation_diag import tilt_yaw                          # noqa: E402
from regbim import metrics                                      # noqa: E402
import r48_common as C                                          # noqa: E402
from r47_common import b1_conditions, B2_INPUTS, YAWS, FLOOR    # noqa: E402

R46 = os.path.expanduser("~/rizu/r46_stsl_data")


def jl(line):
    return json.loads(re.sub(r"(?<![A-Za-z])(-?)inf(?![A-Za-z])", lambda m: m.group(1) + "Infinity", line).replace("nan", "NaN"))


def big(p):
    return (not p["v"]) or p["eR"] >= 45.0 or p["eO"] >= 1.0


def decomp(T, G, om):
    T = np.asarray(T, dtype=np.float64); RT = metrics.decompose_sim3(T)[0]; RG = metrics.decompose_sim3(G)[0]
    ty = tilt_yaw(RT, RG); c = om.mean(0)
    d = metrics.apply_sim3(T, c[None])[0] - metrics.apply_sim3(G, c[None])[0]
    return {"horiz": float(np.hypot(d[0], d[1])), "vert": float(abs(d[2])), "yaw": float(abs(ty["yaw_deg"])), "tilt": float(ty["tilt_deg"])}


def stsl_rep(root, setname, floor, spec, G, om, S, mode):
    """K = 5 の回から (多数決の失敗, 代表の T)"""
    runs = {i: [] for i in range(len(spec["trials"]))}
    for k in range(1, 6):
        trp = json.load(open(os.path.join(root, "processed", "%s_A_k%d" % (setname, k), "30m", "T_rp.json")))
        pt = json.load(open(os.path.join(root, "results", setname, "A_k%d_prep_time.json" % k)))
        for r in (jl(l) for l in open(os.path.join(root, "results", setname, "A_k%d_%s.jsonl" % (k, floor)))):
            i = r["index"]; t = spec["trials"][i]
            Tin = S @ np.asarray(r["tf"]).reshape(4, 4) @ np.asarray(trp[str(i)])
            T = Tin @ np.asarray(t["A"]) @ np.asarray(t["Rg"]) if mode == "ARg" else Tin @ np.asarray(t["P_src_to_input"])
            p = per_trial(T, None, G, om, None)
            if not (len(r["candidates"]) > 0 and isfinite(r["optimal_score"]) and i not in pt.get("prep_failed", [])):
                p = dict(p, v=False)
            runs[i].append((p, T))
    out = []
    for i in range(len(spec["trials"])):
        rs = runs[i]; f = sum(big(p) for p, _ in rs) >= 3
        side = [(p, T) for p, T in rs if big(p) == f]
        side.sort(key=lambda x: x[0]["eO"]); out.append((f, side[len(side) // 2][1], side[len(side) // 2][0]))
    return out


def q(v):
    v = np.asarray(v, dtype=float)
    return {"n": int(len(v)), "p5": float(np.percentile(v, 5)), "median": float(np.median(v)), "p95": float(np.percentile(v, 95))} if len(v) else None


def summarize(rows):
    return {m: {k: q([r[m][k] for r in rows]) for k in ("horiz", "vert", "yaw", "tilt", "eO")} for m in ("planA", "STSL")}


def add(rows, Ta, Ts, G, om):
    a = decomp(Ta, G, om); a["eO"] = per_trial(Ta, None, G, om, None)["eO"]
    s = decomp(Ts, G, om); s["eO"] = per_trial(Ts, None, G, om, None)["eO"]
    rows.append({"planA": a, "STSL": s})


def r48_rows():
    man = json.load(open(os.path.join(C.D, "inputs_manifest.json"))); Gm = np.asarray(man["G"])
    om = dict(np.load(os.path.join(C.D, "omega.npz"))); gta = np.load(os.path.join(C.D, "gta.npz"))
    rows = []
    for ref in C.all_refs():
        scene = C.scene_of(ref); G = gta["G_" + scene] if ref.startswith("GTA_") else Gm; o = om[scene]
        zf = json.load(open(os.path.join(C.D, "BIM", ref, "meta.json")))["z_floor"]; S = np.eye(4); S[2, 3] = zf
        for cut in C.cuts_of(ref):
            setname = "%s__%s" % (ref, cut); spec = json.load(open(os.path.join(C.D, "inputs", setname, "spec.json")))
            if ref.startswith("GTA_"):
                tm = json.load(open("output/Registration/r45/plan_a/%s/trial_matrices.json" % scene))
                rr = {t["trial"]: t for t in tm["trials"] if t["trial"] >= 0}
                TA = [np.asarray(rr[k]["T_est"]) @ np.asarray(rr[k]["P"]) for k in range(C.N_TRIALS)]
            else:
                rr = json.load(open(os.path.join(C.D, "results_sni", setname + ".json")))["rows"]
                TA = [np.asarray(rr["planA|%d" % k]["T_est"]) @ np.asarray(rr["planA|%d" % k]["P"]) for k in range(C.N_TRIALS)]
            st = stsl_rep(C.D, setname, ref, spec, G, o, S, "ARg")
            for k in range(C.N_TRIALS):
                if not big(per_trial(TA[k], None, G, o, None)) and not st[k][0]:
                    add(rows, TA[k], st[k][1], G, o)
    return rows


def real_b1_rows():
    om_all = np.load("Registration/output/diag/omega.npz"); rows = []
    conds = [(c["scene"], c["ref"], c["input_set"], c["ref_target"], c["cut"], c["cross"], c["name"]) for c in b1_conditions()]
    for s in ("m3_cor_a", "m3_cor_b", "m3_cor_c", "m3_cor_d"):                     # R46 の r = 8 m（同じ床の組）
        for e, ref in (("E2", "ref411"), ("E3", "ref410_411")):
            conds.append((s, ref, "%s__%s_r8" % (s, e), "%s__%s" % (s, e), "8", False, None))
    for s, ref, input_set, target, cut, cross, name in conds:
        G = real_criteria(s)[1]; om = om_all[s]
        zf = json.load(open(os.path.join(R46, "BIM", FLOOR[ref], "meta.json")))["z_floor"]; S = np.eye(4); S[2, 3] = zf
        setname = "real_" + input_set; spec = json.load(open(os.path.join(R46, "inputs", setname, "spec.json")))
        f = os.path.join(R46, "results_r47", "sni", name + ".json") if cross else os.path.join(R46, "results_sni", "%s_r%s.json" % (target, cut))
        rr = json.load(open(f))["rows"]
        st = stsl_rep(R46, setname, FLOOR[ref] + "_both", spec, G, om, S, "ARg")
        for k in range(10):
            Ta = np.asarray(rr["planA|%d" % k]["T_est"]) @ np.asarray(rr["planA|%d" % k]["P"])
            if not big(per_trial(Ta, None, G, om, None)) and not st[k][0]:
                add(rows, Ta, st[k][1], G, om)
    return rows


def real_b2_rows():
    om_all = np.load("Registration/output/diag/omega.npz"); rows = []
    zf = json.load(open(os.path.join(R46, "BIM", "M3_410_411", "meta.json")))["z_floor"]; S = np.eye(4); S[2, 3] = zf
    for s, cut in B2_INPUTS:
        G = real_criteria(s)[1]; om = om_all[s]; name = "r47rot_%s_%s" % (s, cut)
        spec = json.load(open(os.path.join(R46, "inputs", name, "spec.json")))
        st = stsl_rep(R46, name, "M3_410_411_both", spec, G, om, S, "P")
        rr = json.load(open(os.path.join(R46, "results_r47", "sni", "%s__in410_411__ref410_411_r%s_rot.json" % (s, cut))))["rows"]
        for i in range(len(YAWS)):
            Ta = np.asarray(rr["planA|%d" % i]["T_est"]) @ np.asarray(rr["planA|%d" % i]["P"])
            if not big(per_trial(Ta, None, G, om, None)) and not st[i][0]:
                add(rows, Ta, st[i][1], G, om)
    return rows


def main():
    res = {}
    for name, fn in (("R48_synthetic", r48_rows), ("real_R46_R47_B1", real_b1_rows), ("real_R47_B2", real_b2_rows)):
        rows = fn(); res[name] = summarize(rows)
        print(name, json.dumps(res[name]), flush=True)
    os.makedirs("Registration/output/diag/r48", exist_ok=True)
    json.dump(res, open("Registration/output/diag/r48/r48_posthoc_b.json", "w"), indent=1)


if __name__ == "__main__":
    main()
