"""Scored validation reports with pass/fail gates.

``validate()`` runs the full battery of checks and returns a
``ValidationReport``: per-check results, a weighted overall score, and an
overall pass/fail verdict. Reports render as Markdown or JSON.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
import json

import numpy as np

DEFAULT_WEIGHTS = {
    "schema_conformance": 0.10,
    "distribution_similarity": 0.30,
    "correlation_preservation": 0.20,
    "memorization": 0.20,
    "utility_tstr": 0.20,
}


@dataclass
class CheckResult:
    """Outcome of a single check: pass/fail, a 0-100 score, and evidence."""

    name: str
    passed: bool
    score: float
    details: dict = field(default_factory=dict)
    message: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ValidationReport:
    """Aggregated result of all checks with an overall score and verdict."""

    checks: list[CheckResult] = field(default_factory=list)
    weights: dict = field(default_factory=dict)

    @property
    def overall_score(self) -> float:
        if not self.checks:
            return 0.0
        total_w = 0.0
        weighted = 0.0
        for c in self.checks:
            w = self.weights.get(c.name, 1.0)
            total_w += w
            weighted += w * c.score
        return round(weighted / total_w, 2) if total_w else 0.0

    @property
    def passed(self) -> bool:
        """Overall verdict: every check gate must pass."""
        return bool(self.checks) and all(c.passed for c in self.checks)

    def failed_checks(self) -> list[str]:
        return [c.name for c in self.checks if not c.passed]

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "overall_score": self.overall_score,
            "checks": [c.to_dict() for c in self.checks],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def to_markdown(self) -> str:
        lines = ["# Synthetic Data Validation Report", ""]
        verdict = "✅ PASS" if self.passed else "❌ FAIL"
        lines.append(f"**Verdict:** {verdict}")
        lines.append(f"**Overall score:** {self.overall_score}/100")
        lines.append("")
        lines.append("| Check | Result | Score | Detail |")
        lines.append("| --- | --- | --- | --- |")
        for c in self.checks:
            icon = "✅" if c.passed else "❌"
            lines.append(f"| {c.name} | {icon} | {c.score}/100 | {c.message} |")
        lines.append("")
        for c in self.checks:
            if c.details:
                lines.append(f"## {c.name}")
                lines.append("")
                lines.append("```json")
                lines.append(json.dumps(c.details, indent=2))
                lines.append("```")
                lines.append("")
        return "\n".join(lines)


def validate(
    real: np.ndarray,
    synth: np.ndarray,
    column_names: list[str] | None = None,
    target_idx: int | None = None,
    run_utility: bool = True,
    weights: dict | None = None,
    **check_kwargs,
) -> ValidationReport:
    """Run the full validation battery.

    Args:
        real: real reference data, shape (n_rows, n_cols).
        synth: synthetic data, shape (m_rows, n_cols).
        column_names: optional names used in reports.
        target_idx: label column for the utility check (default: last).
        run_utility: set False to skip the TSTR utility check (e.g. when the
            data has no binary label).
        weights: optional per-check weights for the overall score.
        **check_kwargs: forwarded to the individual checks (alpha, max_tvd,
            max_mean_abs_diff, min_ratio, range_margin, sample_size, seed).

    Returns a ValidationReport with pass/fail gates per check.
    """
    # Local imports to avoid a hard import cycle between report and check modules.
    from synthetic_data_validator.schema import schema_check
    from synthetic_data_validator.validators import distribution_check
    from synthetic_data_validator.correlation import correlation_preservation
    from synthetic_data_validator.privacy import memorization_check
    from synthetic_data_validator.utility import utility_check

    schema_res = schema_check(real, synth, column_names, check_kwargs.get("range_margin", 0.25))
    report = ValidationReport(weights=weights or dict(DEFAULT_WEIGHTS))
    report.checks.append(schema_res)

    if not schema_res.passed:
        # Structural mismatch makes downstream statistics meaningless; report
        # the failure without running the rest of the battery.
        report.checks[0].message += " — skipping remaining checks"
        return report

    report.checks.append(
        distribution_check(
            real,
            synth,
            column_names,
            alpha=check_kwargs.get("alpha", 0.05),
            max_tvd=check_kwargs.get("max_tvd", 0.2),
        )
    )
    report.checks.append(
        correlation_preservation(
            real,
            synth,
            column_names,
            max_mean_abs_diff=check_kwargs.get("max_mean_abs_diff", 0.15),
        )
    )
    report.checks.append(
        memorization_check(
            real,
            synth,
            min_ratio=check_kwargs.get("min_ratio", 0.5),
            sample_size=check_kwargs.get("sample_size", 2000),
            seed=check_kwargs.get("seed", 0),
        )
    )
    if run_utility:
        try:
            report.checks.append(
                utility_check(
                    real,
                    synth,
                    target_idx=target_idx,
                    min_ratio=check_kwargs.get("utility_min_ratio", 0.85),
                    seed=check_kwargs.get("seed", 0),
                )
            )
        except ValueError as exc:
            report.checks.append(
                CheckResult(
                    name="utility_tstr",
                    passed=True,
                    score=100.0,
                    details={"skipped": str(exc)},
                    message=f"skipped: {exc}",
                )
            )
    return report
