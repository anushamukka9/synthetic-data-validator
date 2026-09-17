"""Tests for synthetic_data_validator."""

import json
import subprocess
import sys

import numpy as np
import pytest

from synthetic_data_validator import (
    ks_test,
    chi_square_test,
    total_variation_distance,
    distribution_check,
    correlation_preservation,
    memorization_check,
    utility_check,
    schema_check,
    validate,
)

rng = np.random.default_rng(42)


def make_real(n=800):
    """Correlated gaussian data with a discrete column and a binary label."""
    x1 = rng.normal(0, 1, n)
    x2 = 0.8 * x1 + rng.normal(0, 0.6, n)
    x3 = rng.normal(5, 2, n)
    x4 = rng.choice([0, 1, 2], size=n, p=[0.5, 0.3, 0.2]).astype(float)
    y = ((x1 + x2 + rng.normal(0, 0.5, n)) > 0.5).astype(float)
    return np.column_stack([x1, x2, x3, x4, y])


def make_good_synth(real, noise=0.15):
    """Synthetic data drawn from the same process with mild perturbation."""
    synth = real + rng.normal(0, noise, real.shape)
    # Keep discrete/label columns on valid values.
    synth[:, 3] = np.clip(np.round(synth[:, 3]), 0, 2)
    synth[:, 4] = (synth[:, 4] > 0.5).astype(float)
    return synth


# --- distribution similarity ---


def test_tvd_identical_distributions_is_near_zero():
    col = rng.normal(0, 1, 2000)
    assert total_variation_distance(col, col) < 1e-9


def test_tvd_disjoint_distributions_is_near_one():
    a = rng.normal(0, 1, 2000)
    b = rng.normal(20, 1, 2000)
    assert total_variation_distance(a, b) > 0.95


def test_ks_detects_shifted_distribution():
    a = rng.normal(0, 1, 1000)
    b = rng.normal(0.5, 1, 1000)
    stat, p = ks_test(a, b)
    assert stat > 0.1
    assert p < 0.01


def test_ks_accepts_same_distribution():
    a = rng.normal(0, 1, 1000)
    b = rng.normal(0, 1, 1000)
    stat, p = ks_test(a, b)
    assert p > 0.05


def test_chi_square_accepts_same_discrete():
    t_rng = np.random.default_rng(1234)
    a = t_rng.choice([0, 1, 2], size=2000, p=[0.5, 0.3, 0.2]).astype(float)
    b = t_rng.choice([0, 1, 2], size=2000, p=[0.5, 0.3, 0.2]).astype(float)
    _, p = chi_square_test(a, b)
    assert p > 0.05


def test_chi_square_rejects_different_discrete():
    a = rng.choice([0, 1, 2], size=1000, p=[0.5, 0.3, 0.2]).astype(float)
    b = rng.choice([0, 1, 2], size=1000, p=[0.1, 0.1, 0.8]).astype(float)
    _, p = chi_square_test(a, b)
    assert p < 0.01


def test_distribution_check_passes_for_good_synth():
    real = make_real()
    synth = make_good_synth(real)
    res = distribution_check(real, synth)
    assert res.passed
    assert res.score >= 80.0


def test_distribution_check_fails_for_bad_synth():
    real = make_real()
    synth = rng.normal(10, 0.1, real.shape)  # totally different marginals
    res = distribution_check(real, synth)
    assert not res.passed


# --- correlation ---


def test_correlation_preservation_good_synth():
    real = make_real()
    synth = make_good_synth(real)
    res = correlation_preservation(real, synth)
    assert res.passed
    assert res.details["mean_abs_diff"] < 0.15


def test_correlation_preservation_independent_noise_fails():
    real = make_real()
    synth = rng.normal(0, 1, real.shape)
    res = correlation_preservation(real, synth)
    assert not res.passed


# --- privacy ---


def test_memorization_detects_copied_rows():
    real = make_real(300)
    synth = real.copy()  # blatant memorization
    res = memorization_check(real, synth)
    assert not res.passed
    assert res.details["dcr_ratio"] < 0.5
    assert res.details["exact_duplicate_rows"] == 300


def test_memorization_accepts_independent_draws():
    real = make_real(600)
    fresh = make_real(600)  # same process, independent rows
    res = memorization_check(real, fresh)
    assert res.passed
    assert 0.5 <= res.details["dcr_ratio"] <= 2.0


# --- utility ---


def test_utility_check_good_synth():
    real = make_real(600)
    synth = make_good_synth(real, noise=0.1)
    res = utility_check(real, synth)
    assert res.details["accuracy_ratio"] > 0.8
    assert res.score > 80.0


def test_utility_check_random_labels_fails():
    real = make_real(600)
    synth = real.copy()
    synth[:, 4] = rng.integers(0, 2, size=600).astype(float)  # pure noise label
    res = utility_check(real, synth, min_ratio=0.85)
    assert res.details["accuracy_ratio"] < 0.85


# --- schema ---


def test_schema_check_passes_for_conformant():
    real = make_real()
    synth = make_good_synth(real)
    res = schema_check(real, synth)
    assert res.passed


def test_schema_check_catches_column_mismatch():
    real = make_real()
    res = schema_check(real, real[:, :3])
    assert not res.passed
    assert any("column count" in i for i in res.details["issues"])


def test_schema_check_catches_nan():
    real = make_real()
    synth = make_good_synth(real)
    synth[0, 0] = np.nan
    res = schema_check(real, synth)
    assert not res.passed


# --- full battery ---


def test_validate_end_to_end_passes():
    real = make_real(600)
    synth = make_good_synth(real)
    report = validate(real, synth)
    assert report.passed
    assert report.overall_score > 70.0
    assert len(report.checks) == 5
    md = report.to_markdown()
    assert "PASS" in md
    d = json.loads(report.to_json())
    assert d["passed"] is True


def test_validate_skips_battery_on_schema_failure():
    real = make_real()
    report = validate(real, real[:, :2])
    assert not report.passed
    assert report.failed_checks() == ["schema_conformance"]


# --- CLI ---


def test_cli_exit_zero_on_good_data(tmp_path):
    real = make_real(400)
    synth = make_good_synth(real)
    rp, sp = tmp_path / "real.csv", tmp_path / "synth.csv"
    np.savetxt(rp, real, delimiter=",", header="x1,x2,x3,x4,y", comments="")
    np.savetxt(sp, synth, delimiter=",", header="x1,x2,x3,x4,y", comments="")
    proc = subprocess.run(
        [sys.executable, "-m", "synthetic_data_validator.cli",
         str(rp), str(sp), "--header", "--no-utility"],
        capture_output=True, text=True, cwd="/tmp",
    )
    assert proc.returncode == 0, proc.stderr
    assert "PASS" in proc.stdout


def test_cli_exit_one_on_bad_data(tmp_path):
    real = make_real(400)
    synth = rng.normal(10, 0.1, real.shape)
    rp, sp = tmp_path / "real.csv", tmp_path / "synth.csv"
    np.savetxt(rp, real, delimiter=",")
    np.savetxt(sp, synth, delimiter=",")
    proc = subprocess.run(
        [sys.executable, "-m", "synthetic_data_validator.cli",
         str(rp), str(sp), "--no-utility"],
        capture_output=True, text=True, cwd="/tmp",
    )
    assert proc.returncode == 1, proc.stderr
    assert "FAIL" in proc.stdout
