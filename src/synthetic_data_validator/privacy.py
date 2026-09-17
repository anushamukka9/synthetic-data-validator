"""Memorization detection: a nearest-neighbor privacy proxy.

A generative model that memorizes its training data emits synthetic rows that
sit suspiciously close to real rows. This check compares, for every synthetic
row, the distance to its nearest real neighbor (DCR) against the typical
nearest-neighbor distance *within* the real data. If synthetic rows are
systematically much closer to real rows than real rows are to each other, the
synthetic data likely leaks real records.

All distances are computed on z-score standardized columns so that no single
column dominates. This is a heuristic privacy proxy, not a privacy guarantee.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist

from synthetic_data_validator.report import CheckResult


def _standardize(real: np.ndarray, synth: np.ndarray):
    real = np.asarray(real, dtype=float)
    synth = np.asarray(synth, dtype=float)
    mu = real.mean(axis=0)
    sigma = real.std(axis=0)
    sigma[sigma == 0] = 1.0
    return (real - mu) / sigma, (synth - mu) / sigma


def _nearest_neighbor_distances(points: np.ndarray, others: np.ndarray) -> np.ndarray:
    d = cdist(points, others, metric="euclidean")
    return d.min(axis=1)


def _nearest_neighbor_distances_self(points: np.ndarray) -> np.ndarray:
    d = cdist(points, points, metric="euclidean")
    np.fill_diagonal(d, np.inf)
    return d.min(axis=1)


def memorization_check(
    real: np.ndarray,
    synth: np.ndarray,
    min_ratio: float = 0.5,
    sample_size: int = 2000,
    seed: int = 0,
) -> CheckResult:
    """Detect memorization via the distance-to-closest-record (DCR) ratio.

    ratio = median(DCR synth->real) / median(DCR real->real, excluding self)

    A ratio near 1 means synthetic rows are about as close to real rows as real
    rows are to each other (healthy). A ratio well below 1 means synthetic rows
    hug real rows too closely (memorization). The check passes when
    ratio >= min_ratio. Also counts exact (near-)duplicate synthetic rows of
    real rows.

    For datasets larger than ``sample_size``, rows are subsampled (with a fixed
    seed) to keep the O(n*m) distance matrix tractable.
    """
    real = np.asarray(real, dtype=float)
    synth = np.asarray(synth, dtype=float)
    if real.ndim != 2 or synth.ndim != 2:
        raise ValueError("memorization_check expects 2-D arrays")
    if real.shape[1] != synth.shape[1]:
        raise ValueError("column count mismatch between real and synthetic data")
    rng = np.random.default_rng(seed)
    if real.shape[0] > sample_size:
        real = real[rng.choice(real.shape[0], sample_size, replace=False)]
    if synth.shape[0] > sample_size:
        synth = synth[rng.choice(synth.shape[0], sample_size, replace=False)]
    if real.shape[0] < 2:
        raise ValueError("need at least 2 real rows for the reference distances")

    real_z, synth_z = _standardize(real, synth)
    d_synth_to_real = _nearest_neighbor_distances(synth_z, real_z)
    d_real_to_real = _nearest_neighbor_distances_self(real_z)

    median_synth = float(np.median(d_synth_to_real))
    median_real = float(np.median(d_real_to_real))
    ratio = median_synth / median_real if median_real > 0 else float("inf")
    exact_dupes = int(np.sum(d_synth_to_real < 1e-9))
    dupe_rate = exact_dupes / len(synth_z) if len(synth_z) else 0.0

    ok = bool(ratio >= min_ratio and dupe_rate < 0.01)
    # Score: 100 at ratio >= 1, linearly down to 0 at ratio -> 0.
    score = max(0.0, min(100.0, 100.0 * ratio))
    details = {
        "dcr_ratio": round(float(ratio), 4),
        "median_synth_to_real": round(median_synth, 4),
        "median_real_to_real": round(median_real, 4),
        "exact_duplicate_rows": exact_dupes,
        "exact_duplicate_rate": round(float(dupe_rate), 4),
        "min_ratio": min_ratio,
        "n_real_sampled": int(real.shape[0]),
        "n_synth_sampled": int(synth.shape[0]),
    }
    return CheckResult(
        name="memorization",
        passed=ok,
        score=round(float(score), 2),
        details=details,
        message=(
            f"DCR ratio = {ratio:.3f} (threshold {min_ratio}); "
            f"{exact_dupes} exact-duplicate synthetic rows"
        ),
    )
