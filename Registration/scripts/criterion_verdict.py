"""R18 §2-2 / R17 §5 — 40 条件の合否を、6基準それぞれで判定する。

**「2/40 が成功」と書く前に、それが基準1（現行 GT）の選択に依存していないかを確かめる。**
`m3_block_b E2` の成功余裕は 0.047 m だが、同シーンの基準の広がりは中央値 0.063 m。
**余裕 < 広がりなので、別の幾何基準を採れば不合格になりうる。**

判定（R18 §2-2、**測る前に固定**）：

===========================  ==========================================
全 6 基準で合格               **「成功」と書いてよい**
全 6 基準で不合格             不合格
合格する基準と不合格の基準が混在  **「基準依存」。成功にも不合格にも数えない**
===========================  ==========================================

**指標ごとの最小値を別々の基準から集めない。**
各基準 j について（回転 ∧ 並進 ∧ 縮尺）の AND を取ってから、基準間で比較する。

**並進は R17 §5 の $d_\\Omega$ で測る**：
$\\Omega$ = seed 0 で抽出した source の構造点（wall/floor/ceiling）を固定し、
$d_\\Omega(T,G) = \\mathrm{RMS}_{p\\in\\Omega}\\|Tp - Gp\\|$。
**変換の並進ベクトル差は source 原点の取り方に依存し、重心も被覆で動く**ので使わない。
"""

from __future__ import annotations

import argparse
import glob
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

from failure_decomposition import provenance                  # noqa: E402
from regbim import io_utils                                   # noqa: E402
from regbim.labels import NAME_TO_ID                          # noqa: E402
from regbim.metrics import (apply_sim3, decompose_sim3,       # noqa: E402
                            rotation_error_deg)

# Ω を作る構造クラス。開口（door/window）は面積が小さく、
# ラベル誤りの影響が相対的に大きいので入れない。
OMEGA_CLASSES = ("wall", "floor", "ceiling")
OMEGA_N = 20000            # Ω の点数。固定して保存する


def build_omega(cfg: Dict, seed: int = 0) -> np.ndarray:
    """seed 0 の source から構造点を固定抽出する（R17 §5）。"""
    c = dict(cfg, source=dict(cfg["source"], seed=seed))
    src = io_utils.load_source_cloud(c)
    ids = {NAME_TO_ID[n] for n in OMEGA_CLASSES if n in NAME_TO_ID}
    m = np.isin(src.labels, list(ids))
    pts = src.points[m]
    if len(pts) == 0:
        raise ValueError("Ω が空。構造クラスが source に無い")
    # 決定的に間引く（乱数を使わない）
    if len(pts) > OMEGA_N:
        pts = pts[np.linspace(0, len(pts) - 1, OMEGA_N).astype(int)]
    return np.ascontiguousarray(pts, dtype=np.float64)


def d_omega(T: np.ndarray, G: np.ndarray, omega: np.ndarray) -> float:
    a = apply_sim3(np.asarray(T, dtype=np.float64), omega)
    b = apply_sim3(np.asarray(G, dtype=np.float64), omega)
    return float(np.sqrt(((a - b) ** 2).sum(axis=1).mean()))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--direct", default="Registration/output/diag/realdata_direct_v2.json")
    ap.add_argument("--spread", default="Registration/output/diag/criterion_spread.json")
    ap.add_argument("--omega-cache", default="Registration/output/diag/omega.npz")
    ap.add_argument("--out", default="Registration/output/diag/criterion_verdict.json")
    args = ap.parse_args()

    direct = json.load(open(args.direct))["results"]
    spread = json.load(open(args.spread))["scenes"]

    # Ω を作って保存する（条件をまたいで同じものを使う）
    omegas: Dict[str, np.ndarray] = {}
    if os.path.exists(args.omega_cache):
        z = np.load(args.omega_cache)
        omegas = {k: z[k] for k in z.files}
        print("Ω をキャッシュから読んだ（%d シーン）" % len(omegas))
    scenes = sorted({r["scene"] for r in direct})
    for s in scenes:
        if s in omegas:
            continue
        cfg = yaml.safe_load(open("Registration/configs/realdata/%s__E3.yaml" % s))
        omegas[s] = build_omega(cfg)
        print("  Ω[%s] = %d 点" % (s, len(omegas[s])), flush=True)
    np.savez_compressed(args.omega_cache, **omegas)

    # 各条件の成功閾値（config から読む。条件ごとに同じはずだが確認する）
    out: List[Dict] = []
    counts = {"success": 0, "fail": 0, "criterion_dependent": 0, "no_criteria": 0}
    print("\n%-16s %-16s %-8s %s" % ("target", "判定", "合格基準", "d_Ω の範囲 [m]"))
    for r in sorted(direct, key=lambda x: x["target"]):
        scene = r["scene"]
        cfg = yaml.safe_load(open("Registration/configs/realdata/%s.yaml" % r["target"]))
        th = cfg["eval"]["success"]
        G_list = [(f["id"], np.asarray(f["T"], dtype=np.float64))
                  for f in spread.get(scene, {}).get("results", []) if "T" in f]
        if len(G_list) < 6:
            counts["no_criteria"] += 1
            out.append({"target": r["target"], "verdict": "no_criteria"})
            continue
        # 手法の解を復元する。direct の再実行は行列を保存していないので、
        # ここでは誤差から再構成できない → 行列を持つ v2 出力が必要
        T = r.get("T_est")
        if T is None:
            counts["no_criteria"] += 1
            out.append({"target": r["target"], "verdict": "no_T_est"})
            continue
        T = np.asarray(T, dtype=np.float64)
        per = []
        for jid, G in G_list:
            RG, _, sG = decompose_sim3(G)
            R, _, s = decompose_sim3(T)
            e = {"id": jid,
                 "rot_deg": rotation_error_deg(R, RG),
                 "d_omega": d_omega(T, G, omegas[scene]),
                 "scale_ratio": float(abs(s / sG - 1.0))}
            e["ok"] = bool(e["rot_deg"] < th["rot_deg"]
                           and e["d_omega"] < th["trans"]
                           and e["scale_ratio"] < th["scale_ratio"])
            per.append(e)
        n_ok = sum(p["ok"] for p in per)
        verdict = ("success" if n_ok == len(per) else
                   "fail" if n_ok == 0 else "criterion_dependent")
        counts[verdict] += 1
        ds = [p["d_omega"] for p in per]
        out.append({"target": r["target"], "scene": scene, "verdict": verdict,
                    "n_ok": n_ok, "n_criteria": len(per), "per_criterion": per,
                    "d_omega_min": min(ds), "d_omega_max": max(ds),
                    "thresholds": th})
        print("%-16s %-16s %d/%d      [%.4f, %.4f]"
              % (r["target"], verdict, n_ok, len(per), min(ds), max(ds)))

    print("\n**判定の内訳**")
    for k, lab in (("success", "全基準で合格（成功）"),
                   ("criterion_dependent", "**基準依存**"),
                   ("fail", "全基準で不合格"),
                   ("no_criteria", "判定できず")):
        print("  %-22s %d / %d" % (lab, counts[k], len(direct)))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"provenance": provenance(), "omega_classes": OMEGA_CLASSES,
                   "omega_n": OMEGA_N, "counts": counts, "results": out},
                  f, indent=2, ensure_ascii=False)
    print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
