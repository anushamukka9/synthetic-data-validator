"""Correlation-structure preservation check.

Synthetic data should preserve the relationships between columns, not just the
marginals. This module compares the Pearson correlation matrices of the real
and synthetic datasets and reports how far apart they are.
"""

from __future__ import annotations

import numpy as np

from synthetic_data_validator.report import CheckResult


def _correlation_matrix(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError("expected a 2-D array")
    c = np.corrcoef(x, rowvar=False)
    return np.nan_to_num(c, nan=0.0, posinf=1.0, neginf=-1.0)


def correlation_preservation(
    real: np.ndarray,
    synth: np.ndarray,
    column_names: list[str] | None = None,
    max_mean_abs_diff: float = 0.15,
) -> CheckResult:
    """Compare correlation matrices of real and synthetic data.

    Reports the mean absolute difference and the Frobenius norm of the
    difference between the two correlation matrices. The check passes when the
    mean absolute difference is <= max_mean_abs_diff. Constant columns are
    handled by treating NaN correlations as 0.
    """
    real = np.asarray(real, dtype=float)
    synth = np.asarray(synth, dtype=float)
    if real.shape[1] != synth.shape[1]:
        raise ValueError("column count mismatch between real and synthetic data")
    if real.shape[1] < 2:
        return CheckResult(
            name="correlation_preservation",
            passed=True,
            score=100.0,
            details={"note": "fewer than 2 columns; nothing to compare"},
            message="skipped: fewer than 2 columns",
        )
    cr = _correlation_matrix(real)
    cs = _correlation_matrix(synth)
    diff = np.abs(cr - cs)
    mean_abs = float(np.mean(diff))
    frobenius = float(np.linalg.norm(cr - cs, ord="fro"))
    names = column_names or [f"col_{i}" for i in range(real.shape[1])]
    worst = None
    if names:
        idx = np.unravel_index(int(np.argmax(diff)), diff.shape)
        worst = {
            "pair": (names[idx[0]], names[idx[1]]),
            "real_corr": round(float(cr[idx]), 4),
            "synth_corr": round(float(cs[idx]), 4),
            "abs_diff": round(float(diff[idx]), 4),
        }
    ok = mean_abs <= max_mean_abs_diff
    # Score: 100 at zero difference, decaying to 0 as mean abs diff -> 1.
    score = max(0.0, 100.0 * (1.0 - mean_abs / max(mean_abs, max_mean_abs_diff, 1e-9)))
    details = {
        "mean_abs_diff": round(mean_abs, 4),
        "frobenius_norm": round(frobenius, 4),
        "worst_pair": worst,
        "max_mean_abs_diff": max_mean_abs_diff,
    }
    return CheckResult(
        name="correlation_preservation",
        passed=bool(ok),
        score=round(float(score), 2),
        details=details,
        message=(
            f"mean |corr_real - corr_synth| = {mean_abs:.4f} "
            f"(threshold {max_mean_abs_diff})"
        ),
    )
