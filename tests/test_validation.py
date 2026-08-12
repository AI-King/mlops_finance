"""Tests for the training data contract."""

import pandas as pd
import pytest

from mlops_finance.data import make_data
from mlops_finance.validation import DataValidationError, validate_training_data


def test_valid_training_data_passes() -> None:
    """Synthetic baseline data should satisfy the training contract."""
    frame = make_data(rows=100, seed=7)

    validate_training_data(frame)


def test_missing_required_column_fails() -> None:
    """The trainer should fail early when a required feature is absent."""
    frame = make_data(rows=100, seed=7).drop(columns=["amount"])

    with pytest.raises(DataValidationError, match="Missing required columns: amount"):
        validate_training_data(frame)


def test_null_values_fail() -> None:
    """Nulls are dangerous because models may fail or learn bad patterns."""
    frame = make_data(rows=100, seed=7)
    frame.loc[0, "merchant_risk"] = pd.NA

    with pytest.raises(DataValidationError, match="null values"):
        validate_training_data(frame)


def test_invalid_feature_range_fails() -> None:
    """Financial amount cannot be negative in this training contract."""
    frame = make_data(rows=100, seed=7)
    frame.loc[0, "amount"] = -1

    with pytest.raises(DataValidationError, match="amount must be greater than 0"):
        validate_training_data(frame)


def test_single_label_class_fails() -> None:
    """Training needs both fraud and non-fraud examples."""
    frame = make_data(rows=100, seed=7)
    frame["fraud"] = 0

    with pytest.raises(DataValidationError, match="both classes"):
        validate_training_data(frame)
