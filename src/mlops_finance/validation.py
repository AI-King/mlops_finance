"""Training data validation rules for financial fraud modeling."""

import pandas as pd

from .data import FEATURES

LABEL = "fraud"
REQUIRED_COLUMNS = [*FEATURES, LABEL]


class DataValidationError(ValueError):
    """Raised when training data does not satisfy the expected contract."""


def validate_training_data(frame: pd.DataFrame) -> None:
    """Validate schema and value ranges before model training.

    Complexity: O(n * c), where n is rows and c is checked columns.
    DSA: set membership finds missing columns in roughly O(c) time.
    """
    missing_columns = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise DataValidationError(f"Missing required columns: {missing}")

    if frame.empty:
        raise DataValidationError("Training data is empty")

    if frame[REQUIRED_COLUMNS].isna().any().any():
        raise DataValidationError("Training data contains null values")

    if not (frame["amount"] > 0).all():
        raise DataValidationError("amount must be greater than 0")

    if not frame["merchant_risk"].between(0, 1).all():
        raise DataValidationError("merchant_risk must be between 0 and 1")

    if not frame["customer_age"].between(18, 100).all():
        raise DataValidationError("customer_age must be between 18 and 100")

    if not (frame["velocity_24h"] >= 0).all():
        raise DataValidationError("velocity_24h must be greater than or equal to 0")

    allowed_labels = {0, 1}
    labels = set(frame[LABEL].unique())
    if not labels.issubset(allowed_labels):
        raise DataValidationError("fraud label must contain only 0 or 1")

    if frame[LABEL].nunique() < 2:
        raise DataValidationError("fraud label must contain both classes: 0 and 1")
