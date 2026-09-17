# synthetic-data-validator

Validate synthetic tabular datasets against real reference data. Before you
train a model on synthetic data, ship a synthetic dataset, or claim a
generative model preserves privacy, run it through this battery of checks:

1. **Distribution similarity** — per-column KS / chi-square tests plus total
   variation distance between histograms.
2. **Correlation preservation** — does the synthetic data keep the
   relationships *between* columns (Frobenius/mean-abs difference of the
   correlation matrices)?
3. **Memorization detection** — a nearest-neighbor privacy proxy: if synthetic
   rows sit much closer to real rows than real rows do to each other, the
   generator is memorizing, not generalizing.
4. **Utility (TSTR)** — train-on-synthetic-test-on-real: a logistic regression
   trained on synthetic data should perform nearly as well on real held-out
   data as one trained on real data.
5. **Schema conformance** — column counts, names, numeric types, no stray
   NaN/inf, value ranges within the real data's range.

Every check produces a pass/fail gate and a 0–100 score; the report combines
them into a weighted overall score. The CLI exits non-zero when any gate
fails, so it works as a quality gate in CI.

Author: [Anusha Mukka](https://anushamukka.com)

## Install

```bash
pip install synthetic-data-validator
# or from source:
pip install .
```

Requires Python ≥ 3.9 and only `numpy` + `scipy`.

## Quickstart

```python
import numpy as np
from synthetic_data_validator import validate

real = np.loadtxt("real.csv", delimiter=",")
synth = np.loadtxt("synthetic.csv", delimiter=",")

report = validate(real, synth)
print(report.to_markdown())
print("PASS" if report.passed else "FAIL", report.overall_score)
```

Or run the runnable example:

```bash
python examples/quickstart.py
```

## CLI

```bash
sdv-validate real.csv synthetic.csv --header
sdv-validate real.csv synthetic.csv --target-idx 4 --format json --output report.json
sdv-validate real.csv synthetic.csv --thresholds '{"alpha": 0.01, "max_tvd": 0.25}'
sdv-validate real.csv synthetic.csv --no-utility   # unlabeled data
```

Options:

| Flag | Purpose |
|---|---|
| `--header` | first row of each CSV holds column names |
| `--target-idx` | label column index for the utility check (default: last) |
| `--no-utility` | skip the TSTR utility check (no binary label available) |
| `--thresholds` | JSON overrides: `alpha`, `max_tvd`, `max_mean_abs_diff`, `min_ratio`, `utility_min_ratio`, `range_margin` |
| `--format` | `markdown` (default) or `json` |
| `--output` | write the report to a file |

Exit code 0 = all gates passed, 1 = at least one gate failed.

## API

```python
from synthetic_data_validator import (
    validate,            # full battery -> ValidationReport
    distribution_check,  # KS / chi-square / TVD per column
    correlation_preservation,
    memorization_check,  # DCR ratio privacy proxy
    utility_check,        # train-on-synthetic-test-on-real
    schema_check,
)

report = validate(real, synth, column_names=[...], target_idx=4,
                  run_utility=True, min_ratio=0.5, max_tvd=0.2)
report.passed          # bool: every gate passed
report.overall_score   # weighted 0-100 score
report.failed_checks() # names of failing checks
report.to_markdown()   # or .to_json()
```

See [`docs/usage.md`](docs/usage.md) for a full guide: what each check
measures, how the gates are scored, and how to pick thresholds.

## Architecture

```
src/synthetic_data_validator/
├── __init__.py      public API
├── cli.py           sdv-validate entry point (CSV loading, thresholds, exit codes)
├── report.py        CheckResult / ValidationReport, validate() orchestration
├── validators.py    ks_test, chi_square_test, total_variation_distance, distribution_check
├── correlation.py   correlation-matrix comparison
├── privacy.py       nearest-neighbor distance-ratio memorization check
├── schema.py        structural conformance
└── utility.py       NumPy logistic regression, train-on-synthetic-test-on-real
```

Checks are independent pure functions over NumPy arrays; `report.validate()`
orchestrates them, applies gates, and renders the scored report. No pandas,
no scikit-learn — `numpy` + `scipy` only.

## Notes & limitations

- Input is numeric tabular data; categorical columns must be encoded first.
- The memorization check is a *privacy proxy*, not a privacy guarantee — a
  passing score does not certify differential privacy.
- The utility check trains logistic regression on a binary label; it measures
  whether a simple downstream model transfers, not whether every possible
  model does.

## License

MIT — see [LICENSE](LICENSE). Copyright © 2026 Anusha Mukka.
