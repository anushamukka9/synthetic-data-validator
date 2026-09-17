"""synthetic_data_validator: quality checks for synthetic tabular data.

Validates a synthetic dataset against a real reference dataset across five
dimensions: per-column distribution similarity, correlation-structure
preservation, memorization (privacy proxy), downstream utility, and schema
conformance. Produces a scored report with pass/fail gates.
"""

from synthetic_data_validator.validators import (
    ks_test,
    chi_square_test,
    total_variation_distance,
    distribution_check,
)
from synthetic_data_validator.correlation import correlation_preservation
from synthetic_data_validator.privacy import memorization_check
from synthetic_data_validator.utility import utility_check
from synthetic_data_validator.schema import schema_check
from synthetic_data_validator.report import CheckResult, ValidationReport, validate

__all__ = [
    "ks_test",
    "chi_square_test",
    "total_variation_distance",
    "distribution_check",
    "correlation_preservation",
    "memorization_check",
    "utility_check",
    "schema_check",
    "CheckResult",
    "ValidationReport",
    "validate",
]

__version__ = "0.1.0"
__author__ = "Anusha Mukka"
