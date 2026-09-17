"""Runnable quickstart: validate a synthetic dataset against real reference data.

Generates a small correlated dataset, produces a synthetic copy with mild
noise (a good generator) and a bad one (shifted marginals, broken
correlations), then runs the full validation battery on both and prints the
reports.
"""

import numpy as np

from synthetic_data_validator import validate

rng = np.random.default_rng(7)


def main() -> None:
    n = 2000
    x1 = rng.normal(0, 1, n)
    x2 = 0.7 * x1 + rng.normal(0, 0.7, n)
    x3 = rng.normal(100, 15, n)
    x4 = rng.choice([0, 1, 2], size=n, p=[0.5, 0.3, 0.2]).astype(float)
    y = ((x1 + x2 + rng.normal(0, 0.4, n)) > 0.4).astype(float)
    real = np.column_stack([x1, x2, x3, x4, y])
    names = ["age_z", "income_z", "spend", "segment", "churn"]

    # A good synthetic generator: fit the real distribution and draw fresh rows.
    mu = real[:, :3].mean(axis=0)
    cov = np.cov(real[:, :3], rowvar=False)
    cont = rng.multivariate_normal(mu, cov, size=n)
    seg = rng.choice([0.0, 1.0, 2.0], size=n,
                     p=[np.mean(x4 == 0), np.mean(x4 == 1), np.mean(x4 == 2)])
    churn = ((cont[:, 0] + cont[:, 1] + rng.normal(0, 0.4, n)) > 0.4).astype(float)
    good = np.column_stack([cont, seg, churn])

    # A bad synthetic generator: column-wise permutation. Marginals and value
    # ranges survive, but every correlation and the label relationship break.
    bad = real.copy()
    for j in range(bad.shape[1]):
        bad[:, j] = rng.permutation(bad[:, j])

    print("=== Good synthetic data ===")
    good_report = validate(real, good, column_names=names)
    print(f"verdict: {'PASS' if good_report.passed else 'FAIL'} "
          f"| score: {good_report.overall_score}/100")

    print("\n=== Bad synthetic data ===")
    bad_report = validate(real, bad, column_names=names)
    print(f"verdict: {'PASS' if bad_report.passed else 'FAIL'} "
          f"| score: {bad_report.overall_score}/100")
    print("failed checks:", ", ".join(bad_report.failed_checks()))

    print("\n=== Markdown report (bad synthetic) ===")
    print(bad_report.to_markdown())


if __name__ == "__main__":
    main()
