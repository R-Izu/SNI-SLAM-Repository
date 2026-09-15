"""R27 §1-3 の検査2・4、および §1-4 の ATE 食い違い。

**`room_0` だけ照合先の配布物が手元に無い。** そこで2通りで攻める。

検査2：`traj.txt` が、手元のどれかの推定軌跡と一致しないか
-----------------------------------------------------------
自作なら、どれかの `estimate_c2w_list`（あるいはそれを変換したもの）と一致するはず。
**生の差だけでなく、Umeyama で合わせた後の残差も見る**
（座標系を変えて保存した可能性があるため。生の差だけでは見逃す）。

検査4：軌跡そのものの性質
--------------------------
**`traj.txt` が配布物と一致した 7 シーンを対照群として使う。**
`room_0` がその 7 本と同じ性質を示すなら、同じ出所である可能性が高い。
**違う性質を示すなら、疑う理由になる。**

見る量：フレーム間の並進の差分（速度）と、その差分（加速度）。
**SLAM の推定軌跡は最適化由来の細かい揺れを持つので、加速度の分布が重い。**
仮想カメラの軌道は滑らかである。

    python Registration/scripts/_r27_traj_probe.py
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "Registration"))
os.chdir(REPO)

from regbim.scale import umeyama   # noqa: E402

SCENES = ["room_0", "room_1", "room_2", "office_0", "office_1",
          "office_2", "office_3", "office_4"]
VERIFIED = set(SCENES) - {"room_0"}      # 配布物とハッシュ一致を確認済み


def traj(scene):
    return np.loadtxt("data/replica/%s_official/traj.txt" % scene).reshape(-1, 4, 4)


# --------------------------------------------------------------------- 検査2
print("=== 検査2：room_0 の traj.txt が、手元のどれかの推定軌跡と一致しないか ===")
t0 = traj("room_0")[:, :3, 3]
ck_paths = sorted(glob.glob("output/Replica/*/*/ckpts/*.tar"))
print("照合先の ckpt: %d 個" % len(ck_paths))
print("%-52s %12s %14s" % ("ckpt", "生の最大差[m]", "Umeyama 残差[m]"))
worst = []
for p in ck_paths:
    try:
        ck = torch.load(p, map_location="cpu")
        e = np.asarray(ck["estimate_c2w_list"], dtype=np.float64)[:, :3, 3]
    except Exception as ex:
        print("%-52s 読めない: %s" % (p[-50:], type(ex).__name__))
        continue
    n = min(len(e), len(t0))
    raw = float(np.abs(e[:n] - t0[:n]).max())
    R, t, s = umeyama(e[:n], t0[:n], with_scaling=True)
    res = float(np.sqrt((((e[:n] @ (s * R).T + t) - t0[:n]) ** 2).sum(axis=1).mean()))
    worst.append((res, raw, p))
    print("%-52s %12.4f %14.4f" % (p[-50:], raw, res))
worst.sort()
if worst:
    print("\n**いちばん近い推定軌跡でも Umeyama 残差 %.4f m**（%s）" % (worst[0][0], worst[0][2]))
    print("  → 0 に近ければ「traj.txt はその推定軌跡から作られた」を疑う")
    print("  → SLAM の ATE（0.44 m）と同程度なら、**別物である**")

# --------------------------------------------------------------------- 検査4
print("\n=== 検査4：軌跡の滑らかさ。配布物と一致した 7 シーンを対照群にする ===")
print("%-10s %-10s %10s %10s %10s %10s %10s"
      % ("シーン", "素性", "速度中央", "速度最大", "加速度中央", "加速度 p99", "加速度最大"))
print("-" * 82)
rows = []
for s in SCENES:
    p = traj(s)[:, :3, 3]
    v = np.linalg.norm(np.diff(p, axis=0), axis=1)          # フレーム間の移動量
    a = np.linalg.norm(np.diff(p, n=2, axis=0), axis=1)     # その差分
    tag = "配布物一致" if s in VERIFIED else "**未照合**"
    rows.append({"scene": s, "verified": s in VERIFIED,
                 "v_median": float(np.median(v)), "v_max": float(v.max()),
                 "a_median": float(np.median(a)), "a_p99": float(np.percentile(a, 99)),
                 "a_max": float(a.max())})
    print("%-10s %-10s %10.5f %10.5f %10.6f %10.6f %10.6f"
          % (s, tag, np.median(v), v.max(), np.median(a),
             np.percentile(a, 99), a.max()))

ctrl = [r for r in rows if r["verified"]]
r0 = [r for r in rows if not r["verified"]][0]
print("\n対照群（7 シーン）の範囲と room_0 の位置：")
for k in ("v_median", "v_max", "a_median", "a_p99", "a_max"):
    lo = min(r[k] for r in ctrl)
    hi = max(r[k] for r in ctrl)
    inside = lo <= r0[k] <= hi
    print("  %-10s 対照群 [%.6f, %.6f]   room_0 %.6f  %s"
          % (k, lo, hi, r0[k], "**範囲内**" if inside else "**範囲外**"))

# 比較のため、推定軌跡（＝SLAM 由来）の同じ量も出す
try:
    ck = torch.load("output/Replica/room0_official/260310_test4/ckpts/01999.tar",
                    map_location="cpu")
    e = np.asarray(ck["estimate_c2w_list"], dtype=np.float64)[:, :3, 3]
    v = np.linalg.norm(np.diff(e, axis=0), axis=1)
    a = np.linalg.norm(np.diff(e, n=2, axis=0), axis=1)
    print("\n**参考：SLAM の推定軌跡（＝自作するならこれが材料）**")
    print("  %-10s %10s %10s %10s %10s %10s" % ("", "速度中央", "速度最大",
                                                "加速度中央", "加速度 p99", "加速度最大"))
    print("  %-10s %10.5f %10.5f %10.6f %10.6f %10.6f"
          % ("推定軌跡", np.median(v), v.max(), np.median(a),
             np.percentile(a, 99), a.max()))
    print("  → **推定軌跡の加速度が traj.txt より明らかに重いなら、traj.txt は推定由来ではない**")
except Exception as ex:
    print("参考の推定軌跡が読めない: %s" % ex)

# --------------------------------------------------------------- §1-4 ATE
print("\n=== §1-4：同じ room0_official の2つの ATE は同じ量か ===")
for p in sorted(glob.glob("output/Replica/room0_official/*/eval_ate.json")):
    try:
        d = json.load(open(p))
    except Exception:
        continue
    print("  %-52s rmse %8.3f %s  n=%s  ckpt=%s"
          % (p[-50:], d.get("absolute_translational_error.rmse", float("nan")),
             d.get("units", "?"), d.get("compared_pose_pairs"),
             os.path.basename(str(d.get("ckpt_path")))))

out = "Registration/output/diag/r27_traj_probe.json"
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w") as f:
    json.dump({"smoothness": rows,
               "closest_estimate": ({"residual_m": worst[0][0], "raw_max_m": worst[0][1],
                                     "ckpt": worst[0][2]} if worst else None)},
              f, indent=2, ensure_ascii=False)
print("\nwrote %s" % out)
