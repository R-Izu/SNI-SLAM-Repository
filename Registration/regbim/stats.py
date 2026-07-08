"""Statistical helpers for benchmark reporting.

Adds the aggregates required by the multi-scene evaluation protocol:
Wilson 95% CI on success rates, error-distribution percentiles, and a
per-criterion failure breakdown (which threshold was exceeded, by how much).
Pure numpy — no scipy dependency.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

# Keys of a per-trial error dict, in reporting order.
ERROR_KEYS: Tuple[str, str, str] = ("rot_deg", "trans", "scale_ratio")


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion k/n (default 95%)."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1.0 - p) / n + z * z / (4 * n * n)) / denom
    return (float(center - half), float(center + half))


def error_percentiles(values: Sequence[float]) -> Dict[str, float]:
    """min / q25 / median / q75 / max plus mean and std of an error sample."""
    if len(values) == 0:
        nan = float("nan")
        return {"min": nan, "q25": nan, "med": nan, "q75": nan, "max": nan,
                "mean": nan, "std": nan}
    arr = np.asarray(values, dtype=float)
    q = np.percentile(arr, [0, 25, 50, 75, 100])
    return {"min": float(q[0]), "q25": float(q[1]), "med": float(q[2]),
            "q75": float(q[3]), "max": float(q[4]),
            "mean": float(arr.mean()), "std": float(arr.std())}


def check_success(err: Dict[str, float], thresholds: Dict[str, float]) -> bool:
    """True when every error component is below its threshold."""
    return all(err[k] < thresholds[k] for k in ERROR_KEYS)


def failure_breakdown(errors: List[Dict[str, float]],
                      thresholds: Dict[str, float]) -> Dict:
    """Summarise failed trials: per-criterion violation counts and excesses.

    Returns counts of trials violating each criterion, counts per violated-
    criterion combination (e.g. "rot_deg+trans"), and the median / max excess
    (error minus threshold) per criterion over its violating trials.
    """
    per_criterion: Dict[str, List[float]] = {k: [] for k in ERROR_KEYS}
    combos: Dict[str, int] = {}
    n_failed = 0
    for err in errors:
        violated = [k for k in ERROR_KEYS if err[k] >= thresholds[k]]
        if not violated:
            continue
        n_failed += 1
        combos["+".join(violated)] = combos.get("+".join(violated), 0) + 1
        for k in violated:
            per_criterion[k].append(err[k] - thresholds[k])
    excess = {}
    for k, vals in per_criterion.items():
        excess[k] = {
            "n": len(vals),
            "median_excess": float(np.median(vals)) if vals else None,
            "max_excess": float(np.max(vals)) if vals else None,
        }
    return {"n_failed": n_failed, "by_criterion": excess, "combinations": combos}
