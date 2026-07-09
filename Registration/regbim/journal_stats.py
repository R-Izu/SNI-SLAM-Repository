"""Journal-level statistical post-processing on per-trial benchmark data.

Everything here consumes the per-trial records saved by benchmark.py
(trials.csv rows) — no experiment re-runs. Provides:
  * alpha-recall curves (success rate vs threshold sweep per error axis)
  * perturbation-magnitude stratified success rates (+ Wilson CI)
  * McNemar's exact test on paired success/failure between two methods
    (valid because every method faces the identical perturbation sequence)
  * Shapiro-Wilk normality tests on error distributions
  * per-trial timing aggregates (mean +/- sigma)
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy import stats as sps

from .stats import wilson_ci

# error axis -> (trials.csv column, sweep range upper bound, unit label)
ALPHA_AXES: Dict[str, Tuple[str, float, str]] = {
    "rot_deg": ("rot_deg", 20.0, "rotation threshold [deg]"),
    "trans": ("trans", 1.0, "translation threshold [m]"),
    "scale_ratio": ("scale_ratio", 0.20, "scale threshold [ratio]"),
}

# perturbation axis -> (column, bin edges, unit label)
PERT_AXES: Dict[str, Tuple[str, np.ndarray, str]] = {
    "pert_rot_deg": ("pert_rot_deg", np.arange(0.0, 181.0, 30.0),
                     "perturbation rotation [deg]"),
    "pert_trans": ("pert_trans", np.linspace(0.0, 3.5, 8),
                   "perturbation |t| [m]"),
    "pert_scale": ("pert_scale", np.array([0.65, 0.8, 0.95, 1.1, 1.25, 1.5]),
                   "perturbation scale [x]"),
}


def alpha_recall(errors: Sequence[float], upper: float,
                 n_points: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    """Recall (fraction of trials with error < alpha) over a threshold sweep."""
    errs = np.asarray(errors, dtype=float)
    alphas = np.linspace(0.0, upper, n_points)
    recall = (errs[None, :] < alphas[:, None]).mean(axis=1)
    return alphas, recall


def stratified_success(pert_values: Sequence[float], successes: Sequence[bool],
                       edges: np.ndarray) -> List[Dict]:
    """Per-bin success rate with Wilson CI over a perturbation magnitude axis."""
    pv = np.asarray(pert_values, dtype=float)
    sc = np.asarray(successes, dtype=bool)
    rows: List[Dict] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (pv >= lo) & (pv < hi)
        n = int(mask.sum())
        k = int(sc[mask].sum())
        lo_ci, hi_ci = wilson_ci(k, n)
        rows.append({"bin_lo": float(lo), "bin_hi": float(hi), "n": n, "k": k,
                     "rate": (k / n) if n else float("nan"),
                     "ci_lo": lo_ci, "ci_hi": hi_ci})
    return rows


def mcnemar_exact(success_a: Sequence[bool], success_b: Sequence[bool]) -> Dict:
    """Exact McNemar test on paired trials (same perturbation sequence).

    b = trials where A succeeds and B fails; c = the reverse. Under H0 the
    discordant pairs are Binomial(b+c, 0.5); two-sided exact p-value.
    """
    a = np.asarray(success_a, dtype=bool)
    bb = np.asarray(success_b, dtype=bool)
    if a.shape != bb.shape:
        raise ValueError("paired samples must have equal length")
    b = int(np.sum(a & ~bb))
    c = int(np.sum(~a & bb))
    n_disc = b + c
    if n_disc == 0:
        p = 1.0
    else:
        p = min(1.0, 2.0 * sps.binom.cdf(min(b, c), n_disc, 0.5))
    return {"n": int(a.size), "both_success": int(np.sum(a & bb)),
            "both_fail": int(np.sum(~a & ~bb)), "a_only": b, "b_only": c,
            "p_value": float(p)}


def shapiro_wilk(errors: Sequence[float]) -> Dict:
    """Shapiro-Wilk normality test (justifies median/quartile reporting)."""
    errs = np.asarray(errors, dtype=float)
    if errs.size < 3 or np.allclose(errs, errs[0]):
        return {"W": float("nan"), "p_value": float("nan"), "normal_at_5pct": None}
    w, p = sps.shapiro(errs)
    return {"W": float(w), "p_value": float(p), "normal_at_5pct": bool(p >= 0.05)}


def timing_stats(times: Sequence[float]) -> Dict:
    """mean +/- sigma (and median) of per-trial registration time."""
    t = np.asarray(times, dtype=float)
    return {"n": int(t.size), "mean_s": float(t.mean()), "std_s": float(t.std()),
            "median_s": float(np.median(t))}
