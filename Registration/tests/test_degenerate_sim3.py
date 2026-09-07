"""潰れた Sim(3) が成功と判定されないこと（R15 §1）。

T3 の `office_0__cov*_ne__*` 4 cell では ICP のスケールが単調に潰れ（0.53→…→1e-30）、
`decompose_sim3` の `max(det, 1e-18)` により s は厳密に 1e-6 として記録されていた。
このとき `R = M / s` はほぼ零行列で、再直交化して得られる回転は**不定**である。
参照側も同じく潰れるため並進・縮尺誤差が ~0 になり閾値を無条件に通り、
**成功判定が「不定な回転どうしの比較」だけで決まっていた**（94 件が成功と記録）。
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from regbim import metrics, stats  # noqa: E402

THRESHOLDS = {"rot_deg": 5.0, "trans": 0.1, "scale_ratio": 0.02}
FAILS = []


def check(name, cond, detail=""):
    print(("  OK   " if cond else "  FAIL ") + name + (("  " + detail) if detail else ""))
    if not cond:
        FAILS.append(name)


def sim3(R, t, s):
    T = np.eye(4)
    T[:3, :3] = s * R
    T[:3, 3] = t
    return T


def main() -> int:
    rng = np.random.default_rng(0)
    Q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1.0

    print("1. 実測された潰れ方の再現（ICP の最終 s = 1e-30）")
    collapsed = sim3(Q, np.array([1.95, 1.33, 0.31]), 1e-30)
    _, _, s_read = metrics.decompose_sim3(collapsed)
    check("s=1e-30 は 1e-6 として読み出される（1e-18 の床）",
          abs(s_read - 1e-6) < 1e-12, "読み出し s=%.3e" % s_read)
    check("is_degenerate_sim3 が真", metrics.is_degenerate_sim3(collapsed))

    print("2. 潰れた解どうしの比較は、閾値を無条件に通る")
    # 潰れると `t = mu_dst - s*mu_src` の第2項が消えて **t は参照の重心に収束する**。
    # どの候補・どの試行でも同じ点に寄るので、並進誤差が 0 になる。
    # （再現実行での実測：trans = 1.54e-12 m）
    mu_dst = np.array([1.95, 1.33, 0.31])
    Q2, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    if np.linalg.det(Q2) < 0:
        Q2[:, 0] *= -1.0
    # 回転はまったく別物なのに、潰れているせいで見分けがつかない
    other = sim3(Q2, mu_dst, 1e-30)
    check("2つの解の回転は実際には大きく異なる",
          metrics.rotation_error_deg(Q, Q2) > 30.0,
          "真の回転差 %.1f 度" % metrics.rotation_error_deg(Q, Q2))
    e = metrics.sim3_errors(collapsed, other)
    check("並進誤差が閾値を通ってしまう", e["trans"] < THRESHOLDS["trans"],
          "trans=%.3e m" % e["trans"])
    check("縮尺誤差が閾値を通ってしまう", e["scale_ratio"] < THRESHOLDS["scale_ratio"],
          "scale_ratio=%.3e" % e["scale_ratio"])
    check("degenerate フラグが立つ", e["degenerate"] is True)
    check("**それでも成功にはならない（拒否が効く）**",
          stats.check_success(e, THRESHOLDS) is False)

    print("3. 拒否は潰れていない解には影響しない")
    good = sim3(Q, np.array([1.0, 2.0, 3.0]), 0.83)
    near = sim3(Q, np.array([1.02, 2.0, 3.0]), 0.833)
    e_ok = metrics.sim3_errors(near, good)
    check("正常な解は degenerate ではない", e_ok["degenerate"] is False)
    check("正常な解の成功判定は従来どおり通る",
          stats.check_success(e_ok, THRESHOLDS) is True,
          "rot=%.3f度 trans=%.4f m" % (e_ok["rot_deg"], e_ok["trans"]))

    print("4. 境界：s=0.83 と s=1e-4 の間で切り替わる")
    check("s=1e-3 は正常扱い",
          not metrics.is_degenerate_sim3(sim3(Q, np.zeros(3), 1e-3)))
    check("s=1e-5 は潰れ扱い",
          metrics.is_degenerate_sim3(sim3(Q, np.zeros(3), 1e-5)))

    print()
    if FAILS:
        print("FAILED: %s" % ", ".join(FAILS))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
