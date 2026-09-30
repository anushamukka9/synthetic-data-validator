"""Tests for wasserstein_distance and the column-statistics summary module."""

import numpy as np
import pytest

from synthetic_data_validator import (
    wasserstein_distance,
    distribution_check,
    describe,
    column_drift,
    column_stats_check,
)

rng = np.random.default_rng(11)


def make_real(n=800):
    x1 = rng.normal(0, 1, n)
    x2 = 0.8 * x1 + rng.normal(0, 0.6, n)
    x3 = rng.normal(5, 2, n)
    return np.column_stack([x1, x2, x3])


# --- wasserstein ---


def test_wasserstein_identical_is_zero():
    col = rng.normal(0, 1, 1000)
    assert wasserstein_distance(col, col) == pytest.approx(0.0)


def test_wasserstein_grows_with_shift():
    a = rng.normal(0, 1, 2000)
    small = rng.normal(0.5, 1, 2000)
    big = rng.normal(5, 1, 2000)
    assert wasserstein_distance(a, small) < wasserstein_distance(a, big)
    assert wasserstein_distance(a, small) == pytest.approx(0.5, abs=0.15)


def test_wasserstein_rejects_empty():
    with pytest.raises(ValueError):
        wasserstein_distance([], [1.0, 2.0])


def test_distribution_check_reports_wasserstein():
    real = make_real()
    synth = real + rng.normal(0, 0.1, real.shape)
    res = distribution_check(real, synth, column_names=["a", "b", "c"])
    for col in res.details["columns"]:
        assert "wasserstein" in col
        assert col["wasserstein"] >= 0.0


# --- describe ---


def test_describe_stats():
    data = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    rows = describe(data, column_names=["x", "y"])
    assert rows[0]["column"] == "x"
    assert rows[0]["mean"] == pytest.approx(2.0)
    assert rows[0]["min"] == pytest.approx(1.0)
    assert rows[0]["max"] == pytest.approx(3.0)
    assert rows[0]["missing_rate"] == pytest.approx(0.0)


def test_describe_counts_missing():
    data = np.array([[1.0], [np.nan], [3.0]])
    rows = describe(data)
    assert rows[0]["missing_rate"] == pytest.approx(1 / 3, abs=1e-4)
    assert rows[0]["mean"] == pytest.approx(2.0)


def test_describe_rejects_1d():
    with pytest.raises(ValueError):
        describe(np.array([1.0, 2.0]))


# --- drift ---


def test_column_drift_ranks_shifted_column_first():
    real = make_real()
    synth = real.copy()
    synth[:, 1] = synth[:, 1] + 5.0  # shift one column hard
    rows = column_drift(real, synth, column_names=["a", "b", "c"])
    assert rows[0]["column"] == "b"
    assert rows[0]["mean_shift"] > 1.0
    # Unshifted columns barely drift.
    rest = {r["column"]: r for r in rows[1:]}
    assert rest["a"]["mean_shift"] < 0.2
    assert rest["c"]["mean_shift"] < 0.2


def test_column_drift_catches_std_change():
    real = make_real()
    synth = real.copy()
    synth[:, 0] = synth[:, 0] * 3.0  # triple the spread, same mean
    rows = column_drift(real, synth)
    first = rows[0]
    assert first["column"] == "col_0"
    assert first["std_ratio"] == pytest.approx(3.0, rel=0.1)


def test_column_drift_range_overlap():
    real = np.array([[0.0], [1.0], [2.0]])
    synth = np.array([[1.0], [2.0], [3.0]])
    rows = column_drift(real, synth)
    # Overlap [1, 2] has length 1; union [0, 3] has length 3.
    assert rows[0]["range_overlap"] == pytest.approx(1 / 3, abs=1e-4)


def test_column_drift_mismatched_columns_raises():
    with pytest.raises(ValueError):
        column_drift(make_real(), make_real()[:, :2])


# --- check ---


def test_column_stats_check_passes_for_good_synth():
    real = make_real()
    synth = real + rng.normal(0, 0.1, real.shape)
    res = column_stats_check(real, synth, column_names=["a", "b", "c"])
    assert res.name == "column_statistics"
    assert res.passed
    assert res.details["flagged_columns"] == []


def test_column_stats_check_flags_drifted_column():
    real = make_real()
    synth = real.copy()
    synth[:, 2] = synth[:, 2] + 10.0
    res = column_stats_check(real, synth, column_names=["a", "b", "c"])
    assert not res.passed
    assert res.details["flagged_columns"] == ["c"]
    assert "c" in res.message


def test_column_stats_check_threshold_is_configurable():
    t_rng = np.random.default_rng(2026)
    x1 = t_rng.normal(0, 1, 800)
    x2 = t_rng.normal(5, 2, 800)
    real = np.column_stack([x1, x2])
    synth = real + t_rng.normal(0, 0.3, real.shape)
    loose = column_stats_check(real, synth, max_mean_shift=10.0)
    tight = column_stats_check(real, synth, max_mean_shift=0.0001)
    assert loose.passed
    assert not tight.passed
