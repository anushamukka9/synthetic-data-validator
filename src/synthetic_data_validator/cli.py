"""Command-line interface: validate a synthetic CSV against a real CSV.

Usage:
    sdv-validate real.csv synthetic.csv
    sdv-validate real.csv synthetic.csv --target-idx 4 --format markdown --output report.md

Exit code is 0 when all gates pass and 1 when any check fails, so the CLI can
be used as a quality gate in CI. Both CSVs must be headerless or headed numeric
tables; use --header to indicate the first row holds column names.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from synthetic_data_validator.report import validate

__version__ = "0.1.0"


def load_csv(path: str, header: bool) -> tuple[np.ndarray, list[str] | None]:
    try:
        if header:
            with open(path) as f:
                names = [h.strip() for h in f.readline().split(",")]
            data = np.genfromtxt(path, delimiter=",", skip_header=1)
        else:
            data = np.genfromtxt(path, delimiter=",")
            names = None
    except (OSError, ValueError) as exc:
        raise SystemExit(f"error reading {path}: {exc}") from exc
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if not np.all(np.isfinite(data)):
        raise SystemExit(f"error: {path} contains non-numeric or missing values")
    if data.shape[0] < 2:
        raise SystemExit(f"error: {path} needs at least 2 data rows")
    return data, names


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="sdv-validate",
        description="Validate a synthetic CSV against a real reference CSV.",
    )
    p.add_argument("real_csv", help="path to the real reference CSV")
    p.add_argument("synthetic_csv", help="path to the synthetic CSV")
    p.add_argument("--header", action="store_true",
                   help="first row of each CSV holds column names")
    p.add_argument("--target-idx", type=int, default=None,
                   help="label column index for the utility check (default: last column)")
    p.add_argument("--no-utility", action="store_true",
                   help="skip the train-on-synthetic-test-on-real utility check")
    p.add_argument("--thresholds", type=str, default=None,
                   help="JSON string overriding check thresholds, e.g. "
                        '\'{"alpha": 0.01, "max_tvd": 0.25}\' '
                        "(keys: alpha, max_tvd, max_mean_abs_diff, min_ratio, "
                        "utility_min_ratio, range_margin)")
    p.add_argument("--format", choices=["markdown", "json"], default="markdown",
                   help="report format (default: markdown)")
    p.add_argument("--output", type=str, default=None,
                   help="write the report to this file instead of stdout")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    real, names = load_csv(args.real_csv, args.header)
    synth, synth_names = load_csv(args.synthetic_csv, args.header)
    if names is None and synth_names is None:
        column_names = None
    elif names and synth_names and names != synth_names:
        print("warning: column names differ between the two files", file=sys.stderr)
        column_names = synth_names
    else:
        column_names = names or synth_names

    thresholds: dict = {}
    if args.thresholds:
        try:
            thresholds = json.loads(args.thresholds)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"error: --thresholds is not valid JSON: {exc}") from exc

    report = validate(
        real,
        synth,
        column_names=column_names,
        target_idx=args.target_idx,
        run_utility=not args.no_utility,
        **thresholds,
    )
    output = report.to_markdown() if args.format == "markdown" else report.to_json()
    if args.output:
        with open(args.output, "w") as f:
            f.write(output + "\n")
        print(f"wrote report to {args.output}")
    else:
        print(output)
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
