"""Utility check: train-on-synthetic, test-on-real (TSTR).

The standard practical test of synthetic data quality: does a model trained on
the synthetic data still perform on real held-out data? This module trains a
plain logistic regression (NumPy implementation, no extra dependencies) on
the real data and on the synthetic data, evaluates both on a real test split,
and reports the ratio of accuracies.
"""

from __future__ import annotations

import numpy as np

from synthetic_data_validator.report import CheckResult


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -30, 30)
    return 1.0 / (1.0 + np.exp(-z))


def _train_logistic_regression(
    x: np.ndarray, y: np.ndarray, lr: float = 0.1, epochs: int = 500, seed: int = 0
) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    mu, sigma = x.mean(axis=0), x.std(axis=0)
    sigma[sigma == 0] = 1.0
    xs = (x - mu) / sigma
    w = rng.normal(0, 0.01, xs.shape[1])
    b = 0.0
    n = xs.shape[0]
    for _ in range(epochs):
        p = _sigmoid(xs @ w + b)
        err = p - y
        w -= lr * (xs.T @ err) / n
        b -= lr * err.mean()
    return w, b, mu, sigma


def _accuracy(w, b, mu, sigma, x, y) -> float:
    xs = (np.asarray(x, dtype=float) - mu) / sigma
    pred = (_sigmoid(xs @ w + b) >= 0.5).astype(float)
    return float(np.mean(pred == np.asarray(y, dtype=float).ravel()))


def _train_test_split(x, y, test_frac=0.3, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(x))
    n_test = max(1, int(len(x) * test_frac))
    te, tr = idx[:n_test], idx[n_test:]
    return x[tr], x[te], y[tr], y[te]


def utility_check(
    real: np.ndarray,
    synth: np.ndarray,
    target_idx: int | None = None,
    min_ratio: float = 0.85,
    seed: int = 0,
) -> CheckResult:
    """Train-on-synthetic-test-on-real vs train-on-real-test-on-real.

    ``target_idx`` names the label column (defaults to the last column). The
    label must be binary; multiclass targets raise ValueError. Trains logistic
    regression on real and on synthetic train data, evaluates accuracy on a
    held-out real test split, and reports the ratio
    acc_synthetic_model / acc_real_model. Passes when ratio >= min_ratio.
    """
    real = np.asarray(real, dtype=float)
    synth = np.asarray(synth, dtype=float)
    if real.shape[1] != synth.shape[1]:
        raise ValueError("column count mismatch between real and synthetic data")
    target_idx = real.shape[1] - 1 if target_idx is None else target_idx
    y_real, y_synth = real[:, target_idx], synth[:, target_idx]
    classes = np.unique(y_real)
    if classes.size != 2:
        raise ValueError(
            f"utility_check needs a binary target; found {classes.size} classes"
        )
    # Normalize labels to 0/1.
    y_real = (y_real == classes[1]).astype(float)
    y_synth = ((synth[:, target_idx] == classes[1]).astype(float))

    xr_tr, xr_te, yr_tr, yr_te = _train_test_split(real, y_real, seed=seed)
    xs_tr = synth[:, :target_idx] if target_idx else synth
    if target_idx is None:
        xs_tr = synth
    else:
        xs_tr = np.delete(synth, target_idx, axis=1)
    xfeat_real_tr = np.delete(xr_tr, target_idx, axis=1)
    xfeat_real_te = np.delete(xr_te, target_idx, axis=1)

    w_real, b_real, mu_r, sg_r = _train_logistic_regression(xfeat_real_tr, yr_tr, seed=seed)
    w_synth, b_synth, mu_s, sg_s = _train_logistic_regression(xs_tr, y_synth, seed=seed)
    acc_real = _accuracy(w_real, b_real, mu_r, sg_r, xfeat_real_te, yr_te)
    acc_synth = _accuracy(w_synth, b_synth, mu_s, sg_s, xfeat_real_te, yr_te)

    ratio = acc_synth / acc_real if acc_real > 0 else 0.0
    ok = bool(ratio >= min_ratio)
    score = max(0.0, min(100.0, 100.0 * ratio))
    details = {
        "target_column": int(target_idx),
        "acc_train_on_real": round(acc_real, 4),
        "acc_train_on_synthetic": round(acc_synth, 4),
        "accuracy_ratio": round(float(ratio), 4),
        "min_ratio": min_ratio,
    }
    return CheckResult(
        name="utility_tstr",
        passed=ok,
        score=round(float(score), 2),
        details=details,
        message=(
            f"TSTR accuracy {acc_synth:.3f} vs TRTR {acc_real:.3f} "
            f"(ratio {ratio:.3f}, threshold {min_ratio})"
        ),
    )
