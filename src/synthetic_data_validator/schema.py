"""Schema and type conformance checks.

Verifies the synthetic dataset is structurally compatible with the real
reference data: same number of columns, same column names, numeric dtypes, no
unexpected missing/infinite values, and value ranges that stay within a sane
margin of the real data.
"""

from __future__ import annotations

import numpy as np

from synthetic_data_validator.report import CheckResult


def schema_check(
    real: np.ndarray,
    synth: np.ndarray,
    column_names: list[str] | None = None,
    range_margin: float = 0.25,
) -> CheckResult:
    """Check structural conformance of synthetic data against the real schema.

    Checks:
      1. same number of columns;
      2. same column names (when provided);
      3. synthetic values are finite numerics;
      4. no NaN in synthetic where the real column has none;
      5. each synthetic column's min/max stays within the real column's
         min/max expanded by ``range_margin`` of the real range
         (constant real columns must be matched by constant synthetic ones).
    """
    real = np.asarray(real)
    synth = np.asarray(synth)
    issues: list[str] = []
    col_details = []

    if real.ndim != 2 or synth.ndim != 2:
        issues.append("both datasets must be 2-D arrays")

    if real.shape[1] != synth.shape[1]:
        issues.append(
            f"column count mismatch: real={real.shape[1]}, synthetic={synth.shape[1]}"
        )
    if column_names and len(column_names) != synth.shape[1]:
        issues.append(
            f"{len(column_names)} column names provided for "
            f"{synth.shape[1]} synthetic columns"
        )
    names = (
        column_names
        if column_names
        else [f"col_{i}" for i in range(synth.shape[1])]
    )

    n_cols = min(real.shape[1], synth.shape[1]) if real.ndim == 2 else 0
    for j in range(n_cols):
        r = np.asarray(real[:, j])
        s = np.asarray(synth[:, j])
        col_issues = []
        try:
            s = s.astype(float)
        except (TypeError, ValueError):
            col_issues.append("non-numeric values")
            s = None
        if s is not None:
            if not np.all(np.isfinite(s)):
                col_issues.append("contains NaN or infinite values")
            if np.all(np.isfinite(r.astype(float))) and not np.all(
                np.isfinite(np.asarray(s, dtype=float))
            ):
                col_issues.append("NaN/inf where real column is complete")
            r_clean = r.astype(float)[np.isfinite(r.astype(float))]
            s_clean = s[np.isfinite(s)]
            if r_clean.size and s_clean.size:
                r_min, r_max = r_clean.min(), r_clean.max()
                r_range = r_max - r_min
                if r_range == 0:
                    if not np.allclose(s_clean, r_min):
                        col_issues.append(
                            f"real column is constant ({r_min}); synthetic varies"
                        )
                else:
                    lo = r_min - range_margin * r_range
                    hi = r_max + range_margin * r_range
                    out = np.sum((s_clean < lo) | (s_clean > hi))
                    if out:
                        col_issues.append(
                            f"{out}/{s_clean.size} values outside "
                            f"real range +/- {range_margin*100:.0f}%"
                        )
        col_details.append({"column": names[j], "issues": col_issues})
        issues.extend(f"{names[j]}: {c}" for c in col_issues)

    passed = not issues
    score = 100.0 if passed else max(0.0, 100.0 - 25.0 * len(issues))
    return CheckResult(
        name="schema_conformance",
        passed=passed,
        score=round(float(score), 2),
        details={
            "issues": issues,
            "columns": col_details,
            "range_margin": range_margin,
        },
        message="schema conformance OK" if passed else f"{len(issues)} schema issue(s)",
    )
