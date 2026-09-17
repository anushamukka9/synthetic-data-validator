# Usage guide

## What the checks measure

### 1. Schema conformance (weight 0.10)

Structural sanity: same column count, matching column names (when given),
numeric dtypes, no NaN/inf, and synthetic value ranges within the real
column's min/max expanded by `range_margin` (default 25%). If this check
fails, the remaining battery is skipped — statistics on structurally
incompatible data would be meaningless.

### 2. Distribution similarity (weight 0.30)

For each column:

- Continuous columns (>10 unique values in the real data): two-sample
  Kolmogorov–Smirnov test. Passes when p ≥ `alpha` (default 0.05).
- Discrete columns: chi-square homogeneity test on shared quantile bins.
- Every column also gets a total variation distance (TVD) between histograms,
  a scale-free number in [0, 1]; passes when TVD ≤ `max_tvd` (default 0.2).

A column passes only if **both** its hypothesis test and its TVD gate pass.
The check passes when every column passes; the score is the fraction of
columns passing × 100.

### 3. Correlation preservation (weight 0.20)

Compares the Pearson correlation matrices of real and synthetic data.
Passes when the mean absolute element-wise difference ≤
`max_mean_abs_diff` (default 0.15). The report also shows the Frobenius norm
of the difference and the worst-matching column pair. Constant columns are
treated as zero correlation rather than NaN.

### 4. Memorization (weight 0.20)

For each synthetic row, the distance to its nearest real neighbor (DCR) is
computed on z-score-standardized columns; the same is done for real rows
against other real rows. The reported **DCR ratio** is

```
median(DCR synth → real) / median(DCR real → real, excluding self)
```

- Ratio ≈ 1: healthy — synthetic rows are as close to real rows as real rows
  are to each other.
- Ratio ≪ 1: memorization — synthetic rows hug real records too closely.

Passes when the ratio ≥ `min_ratio` (default 0.5) **and** fewer than 1% of
synthetic rows are exact duplicates of real rows. Large datasets are
subsampled (default 2000 rows, fixed seed) to keep the distance matrix
tractable. This is a heuristic privacy proxy, not a privacy guarantee.

### 5. Utility — train on synthetic, test on real (weight 0.20)

A plain logistic regression (NumPy implementation) is trained on the real
training data and on the synthetic data; both are evaluated on a held-out
real test split. The reported **accuracy ratio** is
`acc(synthetic-trained) / acc(real-trained)`. Passes when the ratio ≥
`utility_min_ratio` (default 0.85). Needs a binary label column
(`target_idx`, default: last column); if the target is not binary, the check
is skipped with a note rather than failing.

## Overall score and verdict

Each check scores 0–100 and has a pass/fail gate. The overall score is the
weighted average (weights above, overridable via `validate(..., weights={...})`).
The verdict is **PASS only if every gate passes** — a high average cannot
mask a failed check.

## Choosing thresholds

Tighter thresholds (smaller `max_tvd`/`max_mean_abs_diff`, larger `alpha`/
`min_ratio`/`utility_min_ratio`) reject more generators. Sensible starting
points are the defaults; tighten `min_ratio` toward 1.0 when the synthetic
data will be released publicly, and loosen `utility_min_ratio` when the
downstream task is exploratory rather than production.

## CI integration

```yaml
- run: sdv-validate data/real.csv data/synthetic.csv --header
```

The step fails (exit 1) when any gate fails, blocking merges of low-quality
synthetic data.
