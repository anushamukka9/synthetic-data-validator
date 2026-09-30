"""Per-column descriptive-statistics comparison.

Hypothesis tests and correlation matrices answer "is the synthetic data
statistically close?", but when a check fails you want to know *where*.
This module computes, for every column, the basic descriptive statistics of
the real and synthetic data side by side (mean, std, min, max, median,
missing rate) and ranks columns by relative drift, so a failing
``distribution_similarity`` gate turns into an actionable to-fix list.

This is a diagnostic check, not a gate: it is exported for standalone use
and is deliberately not part of the default ``validate()`` battery.
"""

from __future__ import annotations

import numpy as np

from synthetic_data_validator.report import CheckResult


def _col_stats(col: np.ndarray) -> dict:
    arr = np.asarray(col, dtype=float).ravel()
    missing = int(np.sum(~np.isfinite(arr)))
    clean = arr[np.isfinite(arr)]
    if clean.size == 0:
        return {
            "n": int(arr.size),
            "missing_rate": 1.0,
            "mean": None, "std": None, "min": None, "max": None,
            "median": None,
        }
    return {
        "n": int(arr.size),
        "missing_rate": round(missing / arr.size, 4),
        "mean": round(float(clean.mean()), 4),
        "std": round(float(clean.std()), 4),
        "min": round(float(clean.min()), 4),
        "max": round(float(clean.max()), 4),
        "median": round(float(np.median(clean)), 4),
    }


def describe(data: np.ndarray, column_names: list[str] | None = None) -> list[dict]:
    """Descriptive statistics for every column of one dataset."""
    data = np.asarray(data, dtype=float)
    if data.ndim != 2:
        raise ValueError("describe expects a 2-D array")
    names = column_names or [f"col_{i}" for i in range(data.shape[1])]
    rows = []
    for j, name in enumerate(names):
        stats = _col_stats(data[:, j])
        stats["column"] = name
        rows.append(stats)
    return rows


def column_drift(
    real: np.ndarray,
    synth: np.ndarray,
    column_names: list[str] | None = None,
) -> list[dict]:
    """Per-column drift between real and synthetic data, worst first.

    For each statistic the relative drift is reported:

    - ``mean_shift``: |mean_s - mean_r| / std_r (standardized, unit-free).
      Columns where the real data is constant use the absolute difference.
    - ``std_ratio``: std_s / std_r (1.0 = identical spread).
    - ``range_overlap``: overlap of [min, max] intervals / union of intervals.
    - ``missing_rate_diff``: missing_s - missing_r.

    The ranking key is ``mean_shift`` (then ``|1 - std_ratio|``), so the
    first rows are the columns a generator most needs to fix.
    """
    real = np.asarray(real, dtype=float)
    synth = np.asarray(synth, dtype=float)
    if real.ndim != 2 or synth.ndim != 2:
        raise ValueError("column_drift expects 2-D arrays")
    if real.shape[1] != synth.shape[1]:
        raise ValueError(
            f"column count mismatch: real has {real.shape[1]}, "
            f"synthetic has {synth.shape[1]}"
        )
    names = column_names or [f"col_{i}" for i in range(real.shape[1])]
    rows = []
    for j, name in enumerate(names):
        r_raw = np.asarray(real[:, j], dtype=float)
        s_raw = np.asarray(synth[:, j], dtype=float)
        r = r_raw[np.isfinite(r_raw)]
        s = s_raw[np.isfinite(s_raw)]
        missing_diff = round(
            float(np.mean(~np.isfinite(s_raw))) - float(np.mean(~np.isfinite(r_raw))),
            4,
        )
        if r.size == 0 or s.size == 0:
            rows.append({
                "column": name, "mean_shift": None, "std_ratio": None,
                "range_overlap": None,
                "missing_rate_diff": missing_diff,
                "note": "no finite values in one of the columns",
            })
            continue
        r_std = float(r.std())
        mean_shift = (
            abs(float(s.mean()) - float(r.mean())) / r_std if r_std > 0
            else abs(float(s.mean()) - float(r.mean()))
        )
        std_ratio = float(s.std()) / r_std if r_std > 0 else (1.0 if float(s.std()) == 0 else float("inf"))
        lo, hi = min(float(r.min()), float(s.min())), max(float(r.max()), float(s.max()))
        overlap = (
            max(0.0, min(float(r.max()), float(s.max())) - max(float(r.min()), float(s.min())))
            / (hi - lo) if hi > lo else 1.0
        )
        rows.append({
            "column": name,
            "real": _col_stats(r),
            "synth": _col_stats(s),
            "mean_shift": round(mean_shift, 4),
            "std_ratio": round(std_ratio, 4) if std_ratio != float("inf") else None,
            "range_overlap": round(overlap, 4),
            "missing_rate_diff": missing_diff,
        })
    rows.sort(
        key=lambda row: (
            row["mean_shift"] if row["mean_shift"] is not None else float("inf"),
            abs(1.0 - (row["std_ratio"] or 1.0)),
        ),
        reverse=True,
    )
    return rows


def column_stats_check(
    real: np.ndarray,
    synth: np.ndarray,
    column_names: list[str] | None = None,
    max_mean_shift: float = 1.0,
) -> CheckResult:
    """Diagnostic check: flag columns whose moments drifted badly.

    Passes when every column's standardized mean shift is <=
    ``max_mean_shift``. The details carry the full ranked drift table, so
    this check doubles as the debugging view for a failing
    ``distribution_similarity`` gate.
    """
    rows = column_drift(real, synth, column_names)
    bad = [
        r["column"] for r in rows
        if r["mean_shift"] is not None and r["mean_shift"] > max_mean_shift
    ]
    worst = max(
        (r["mean_shift"] for r in rows if r["mean_shift"] is not None),
        default=0.0,
    )
    score = max(0.0, 100.0 * (1.0 - worst / (max_mean_shift * 4)))
    return CheckResult(
        name="column_statistics",
        passed=not bad,
        score=round(float(score), 2),
        details={
            "columns": rows,
            "flagged_columns": bad,
            "max_mean_shift": max_mean_shift,
        },
        message=(
            "column statistics look consistent"
            if not bad
            else f"{len(bad)} column(s) drifted: {', '.join(bad)}"
        ),
    )
