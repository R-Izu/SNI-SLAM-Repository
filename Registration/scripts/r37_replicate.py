"""R37 §4 — 回転解放（`proposed.rotation_release`）の再現。8 条件 × (S 10 + P 10) 試行 × off/on。

系列
----
S（標本化）：source の標本化 seed 1〜10。摂動なし。期待値 $G_j$
P（摂動）  ：標本化 seed 0。`benchmark.py` と同じ生成器（`default_rng(cfg.eval.seed)` から
             `metrics.random_sim3(rng, cfg.eval.perturb)` を順に 10 回）。source に $P$ を当てる。
             期待値 $G_jP^{-1}$

off と on の対
--------------
on の実行は、off と同じ経路を最後まで通ったあとに1段足すだけである（`proposed.py`）。
そこで **off の解 = on の実行中の `T_before`** とする。別に回した off 実行との一致は
`--check-off` の抜き取り（条件ごとに S・P 各1試行）で確かめる。

採点
----
Ω は `omega.npz`（seed 0 の source 座標）に固定。摂動した source に対する $\\hat T$ は
**$d_\\Omega(\\hat T P, G_j, \\Omega)$** で測る（Ω を摂動後の座標へ移してから比べるのと同じ）。
回転・傾き・ヨー・縮尺も $\\hat T P$ と $G_j$ で比べる。

失敗の規則は `fd92e22`（R36 §4-2）と同じ。`max_iter` は 10。

    conda activate sni-slam
    python Registration/scripts/r37_replicate.py
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from typing import Dict, List

import numpy as np
import yaml

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "Registration"))
sys.path.insert(0, os.path.join(REPO, "Registration", "scripts"))
os.chdir(REPO)

from criterion_verdict import d_omega                        # noqa: E402
from failure_decomposition import provenance                 # noqa: E402
from r36_release import (FLIP_DEG, MAX_ITER, SCALE_COLLAPSE,  # noqa: E402
                         criteria, iter_stats, score_T)
from regbim import io_utils, metrics, preprocess             # noqa: E402
from regbim.methods import get_method                        # noqa: E402

SCENES = ["m3_cor_a", "m3_cor_b", "m3_cor_c", "m3_cor_d"]
CONDS = ["E2", "E3"]
S_SEEDS = list(range(1, 11))
N_P = 10
PRIMARY = 1


def make_cfg(cfg0: Dict, seed: int, release: bool) -> Dict:
    cfg = copy.deepcopy(cfg0)
    cfg["source"] = dict(cfg["source"], seed=seed)
    cfg.setdefault("proposed", {})["translation_init"] = "plan_correlate"
    cfg["proposed"]["rotation_release"] = {"enabled": bool(release), "max_iter": MAX_ITER}
    cfg.setdefault("diagnostics", {})
    cfg["diagnostics"]["record_yaw"] = True
    cfg["diagnostics"]["record_stages"] = True
    return cfg


def one(target, cfg0, series, k, seed, P, spread, omega, th, check_off) -> Dict:
    scene = target.split("__")[0]
    cfg = make_cfg(cfg0, seed, True)
    src = io_utils.load_source_cloud(cfg)
    dst = io_utils.load_reference_cloud(cfg)
    if P is not None:
        src = metrics.transform_cloud(src, P)
    Pm = np.eye(4) if P is None else np.asarray(P, dtype=np.float64)

    m = get_method("proposed")
    T_on = np.asarray(m.register(src, dst, cfg), dtype=np.float64)
    rd = m.last_release_diag
    st = m.last_stage_diag
    T_off = np.asarray(rd["T_before"], dtype=np.float64)

    off_check = None
    if check_off:
        m2 = get_method("proposed")
        T_off2 = np.asarray(m2.register(src, dst, make_cfg(cfg0, seed, False)),
                            dtype=np.float64)
        off_check = {"bit_equal": bool(np.array_equal(T_off2, T_off)),
                     "max_abs": float(np.abs(T_off2 - T_off).max())}

    crits = criteria(spread, scene)
    # 摂動を戻して、source 座標（Ω の座標）での変換として採点する
    a = score_T(T_off @ Pm, crits, omega, th)
    b = score_T(T_on @ Pm, crits, omega, th)

    # 段階1：正解の向きの種の並進誤差と、選ばれた候補
    G = dict(crits)[PRIMARY]
    RG = metrics.decompose_sim3(G)[0]
    seeds = [np.asarray(t, dtype=np.float64) for t in st["seeds"]]
    rots = [metrics.rotation_error_deg(metrics.decompose_sim3(t @ Pm)[0], RG) for t in seeds]
    kc = int(np.argmin(rots))
    win = int(np.argmax(st["scores"]))

    # 失敗（R36 §4-2 と同じ規則）
    src_p = preprocess.prepare(src, cfg)
    dst_p = preprocess.prepare(dst, cfg)
    its = iter_stats(rd["trace"], src_p, dst_p, cfg)
    _, _, s0 = metrics.decompose_sim3(T_off)
    R0 = metrics.decompose_sim3(T_off)[0]
    R1, _, s1 = metrics.decompose_sim3(T_on)
    dstart = a["decomposition_primary"]["d_omega_m"]
    dend = b["decomposition_primary"]["d_omega_m"]
    rot_change = metrics.rotation_error_deg(R1, R0)
    failures = {
        "scale_collapse": bool(metrics.is_degenerate_sim3(T_on)
                               or not (SCALE_COLLAPSE[0] <= s1 / s0 <= SCALE_COLLAPSE[1])),
        "scale_ratio_end_over_start": float(s1 / s0),
        "flip": bool(rot_change > FLIP_DEG), "rotation_change_deg": float(rot_change),
        "positive_feedback": bool(its[-1]["n_corr"] > its[0]["n_corr"]
                                  and its[-1]["weighted_mean_dist"] < its[0]["weighted_mean_dist"]
                                  and dend > dstart)}
    n_iter = int(rd["trace"][-1][0])
    last_drop = (its[-2]["weighted_mean_dist"] - its[-1]["weighted_mean_dist"]
                 if len(its) >= 2 else None)

    # 旧い採点（摂動前の Ω に T と G P^{-1} を当てる。R30・R36 の GT-A と同じ形）との比較用
    legacy = None
    if P is not None:
        E = G @ metrics.invert_sim3(Pm)
        legacy = {"off": d_omega(T_off, E, omega), "on": d_omega(T_on, E, omega)}

    return {"target": target, "series": series, "k": k, "seed": seed,
            "P": None if P is None else Pm.tolist(),
            "T_off": T_off.tolist(), "T_on": T_on.tolist(),
            "off": a, "on": b,
            "stage1": {"correct_rot_candidate": kc, "correct_rot_err_deg": float(rots[kc]),
                       "seed_d_omega_m": float(d_omega(seeds[kc] @ Pm, G, omega)),
                       "winner": win, "winner_is_correct_rot": bool(win == kc)},
            "failures": failures, "n_iter": n_iter, "hit_max_iter": bool(n_iter >= MAX_ITER),
            "last_iter_residual_drop": last_drop, "off_check": off_check,
            "legacy_d_omega_primary": legacy}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--r35-runs", default="Registration/output/diag/r35_runs.json")
    ap.add_argument("--spread", default="Registration/output/diag/criterion_spread.json")
    ap.add_argument("--omega-cache", default="Registration/output/diag/omega.npz")
    ap.add_argument("--out", default="Registration/output/diag/r37_replicate.json")
    ap.add_argument("--check-off", action="store_true", default=True)
    args = ap.parse_args()

    spread = json.load(open(args.spread))["scenes"]
    z = np.load(args.omega_cache)
    r35 = {("%s__%s" % (r["scene"], r["cond"])): r for r in
           json.load(open(args.r35_runs))["rows"] if r["mode"] == "plan_correlate"}
    res: Dict = {"provenance": provenance(), "max_iter": MAX_ITER, "precheck": [], "rows": []}
    done = set()
    if os.path.exists(args.out):                     # 途中から再開できるようにする
        old = json.load(open(args.out))
        if old.get("provenance", {}).get("commit") == res["provenance"]["commit"]:
            res = old
            done = {(r["target"], r["series"], r["k"]) for r in res["rows"]}

    def dump():
        with open(args.out, "w") as f:
            json.dump(res, f, ensure_ascii=False)

    # §4-4：off・seed 0・摂動なしで R35 の 8 出力とビット一致
    if not res["precheck"]:
        for target, r in r35.items():
            cfg0 = yaml.safe_load(open("Registration/configs/realdata/%s.yaml" % target))
            m = get_method("proposed")
            T = np.asarray(m.register(io_utils.load_source_cloud(make_cfg(cfg0, 0, False)),
                                      io_utils.load_reference_cloud(make_cfg(cfg0, 0, False)),
                                      make_cfg(cfg0, 0, False)), dtype=np.float64)
            eq = bool(np.array_equal(T, np.asarray(r["T_est"])))
            res["precheck"].append({"target": target, "bit_equal": eq})
            print("precheck %-14s bit=%s" % (target, eq), flush=True)
        dump()
        if not all(p["bit_equal"] for p in res["precheck"]):
            print("**precheck 不一致。中止する**")
            return 1

    for scene in SCENES:
        for cond in CONDS:
            target = "%s__%s" % (scene, cond)
            cfg0 = yaml.safe_load(open("Registration/configs/realdata/%s.yaml" % target))
            th = cfg0["eval"]["success"]
            omega = z[scene]
            rng = np.random.default_rng(int(cfg0["eval"]["seed"]))
            Ps = [metrics.random_sim3(rng, cfg0["eval"]["perturb"]) for _ in range(N_P)]
            jobs = ([("S", k, s, None) for k, s in enumerate(S_SEEDS)]
                    + [("P", k, 0, Ps[k]) for k in range(N_P)])
            for series, k, seed, P in jobs:
                if (target, series, k) in done:
                    continue
                row = one(target, cfg0, series, k, seed, P, spread, omega, th,
                          check_off=(args.check_off and k == 0))
                res["rows"].append(row)
                dump()
                a, b = row["off"]["primary"], row["on"]["primary"]
                print("%-14s %s%-2d 回転 %.3f→%.3f dΩ %.4f→%.4f 失敗 %s 反復 %d%s"
                      % (target, series, k, a["rot_deg"], b["rot_deg"],
                         row["off"]["decomposition_primary"]["d_omega_m"],
                         row["on"]["decomposition_primary"]["d_omega_m"],
                         [n for n, v in row["failures"].items() if v is True],
                         row["n_iter"],
                         "" if row["off_check"] is None else
                         " off一致=%s" % row["off_check"]["bit_equal"]), flush=True)
    print("wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
