"""Per-column distribution similarity checks.

Each function compares one column of the real reference data against the same
column of the synthetic data:

- ``ks_test``: two-sample Kolmogorov-Smirnov test (continuous columns).
- ``chi_square_test``: chi-square goodness-of-fit on binned counts (discrete
  columns or binned continuous columns).
- ``total_variation_distance``: half the L1 distance between histograms of the
  two columns, a scale-free similarity measure in [0, 1].
- ``distribution_check``: runs the appropriate checks over every column of the
  datasets and aggregates pass/fail results.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from synthetic_data_validator.report import CheckResult


def _as_float(col) -> np.ndarray:
    arr = np.asarray(col, dtype=float).ravel()
    return arr[~np.isnan(arr)]


def ks_test(real_col, synth_col) -> tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov test.

    Returns (statistic, p_value). A small statistic / large p-value means the
    columns look like they come from the same distribution.
    """
    r, s = _as_float(real_col), _as_float(synth_col)
    if r.size == 0 or s.size == 0:
        raise ValueError("ks_test requires non-empty columns")
    res = stats.ks_2samp(r, s)
    return float(res.statistic), float(res.pvalue)


def chi_square_test(real_col, synth_col, n_bins: int = 10) -> tuple[float, float]:
    """Chi-square test of homogeneity on binned counts.

    Both columns are binned with shared bin edges (quantile edges from the real
    column, so no bin is empty by construction). Returns (statistic, p_value).
    """
    r, s = _as_float(real_col), _as_float(synth_col)
    if r.size == 0 or s.size == 0:
        raise ValueError("chi_square_test requires non-empty columns")
    n_bins = min(int(n_bins), max(2, np.unique(r).size))
    edges = np.quantile(r, np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)
    if edges.size < 3:
        # Degenerate column: everything in one or two bins.
        return 0.0, 1.0
    r_counts, _ = np.histogram(r, bins=edges)
    s_counts, _ = np.histogram(s, bins=edges)
    # Homogeneity test on binned proportions. A pseudocount keeps bins with
    # zero mass from making the test undefined (e.g. disjoint support), and
    # the expected counts are rescaled to match the observed total.
    obs = r_counts.astype(float) + 0.5
    exp = s_counts.astype(float) + 0.5
    exp = exp * (obs.sum() / exp.sum())
    res = stats.chisquare(obs, exp)
    return float(res.statistic), float(res.pvalue)


def total_variation_distance(real_col, synth_col, n_bins: int = 20) -> float:
    """Total variation distance between the two column histograms.

    Returns a value in [0, 1]: 0 for identical histograms, 1 for disjoint
    support. Uses quantile-based binning from the pooled data.
    """
    r, s = _as_float(real_col), _as_float(synth_col)
    if r.size == 0 or s.size == 0:
        raise ValueError("total_variation_distance requires non-empty columns")
    pooled = np.concatenate([r, s])
    n_bins = min(int(n_bins), max(2, np.unique(pooled).size))
    edges = np.quantile(pooled, np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)
    if edges.size < 3:
        return 0.0
    r_hist, _ = np.histogram(r, bins=edges, density=True)
    s_hist, _ = np.histogram(s, bins=edges, density=True)
    widths = np.diff(edges)
    return float(0.5 * np.sum(np.abs(r_hist - s_hist) * widths))


def distribution_check(
    real: np.ndarray,
    synth: np.ndarray,
    column_names: list[str] | None = None,
    alpha: float = 0.05,
    max_tvd: float = 0.2,
) -> CheckResult:
    """Run per-column distribution checks across all columns.

    Continuous columns (more than 10 unique values) use the KS test; discrete
    columns use the chi-square test; every column also gets a total variation
    distance. A column passes if its p-value >= alpha AND its TVD <= max_tvd.
    """
    real = np.asarray(real, dtype=float)
    synth = np.asarray(synth, dtype=float)
    if real.ndim != 2 or synth.ndim != 2:
        raise ValueError("distribution_check expects 2-D arrays")
    if real.shape[1] != synth.shape[1]:
        raise ValueError(
            f"column count mismatch: real has {real.shape[1]}, "
            f"synthetic has {synth.shape[1]}"
        )
    names = column_names or [f"col_{i}" for i in range(real.shape[1])]
    columns = []
    passed = 0
    tvds = []
    for j, name in enumerate(names):
        r, s = _as_float(real[:, j]), _as_float(synth[:, j])
        tvd = total_variation_distance(r, s)
        tvds.append(tvd)
        discrete = np.unique(r).size <= 10
        if discrete:
            stat, p = chi_square_test(r, s)
            test = "chi_square"
        else:
            stat, p = ks_test(r, s)
            test = "ks"
        ok = bool(p >= alpha and tvd <= max_tvd)
        passed += ok
        columns.append(
            {
                "column": name,
                "test": test,
                "statistic": round(stat, 4),
                "p_value": round(float(p), 4),
                "tvd": round(tvd, 4),
                "passed": ok,
            }
        )
    n = len(columns)
    score = 100.0 * passed / n if n else 0.0
    details = {
        "columns": columns,
        "columns_passed": passed,
        "columns_total": n,
        "mean_tvd": round(float(np.mean(tvds)), 4) if tvds else 0.0,
        "alpha": alpha,
        "max_tvd": max_tvd,
    }
    return CheckResult(
        name="distribution_similarity",
        passed=passed == n,
        score=round(score, 2),
        details=details,
        message=f"{passed}/{n} columns passed distribution checks (alpha={alpha}, max_tvd={max_tvd})",
    )
