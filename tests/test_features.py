"""Tests for shared feature engineering."""

import numpy as np
import pandas as pd

from mlops_finance.features import MODEL_FEATURES, build_features


def test_build_features_returns_expected_columns() -> None:
    """Training and serving should receive the same model feature order."""
    frame = pd.DataFrame(
        [
            {
                "amount": 100.0,
                "merchant_risk": 0.8,
                "customer_age": 35,
                "velocity_24h": 4,
            }
        ]
    )

    features = build_features(frame)

    assert list(features.columns) == MODEL_FEATURES


def test_build_features_calculates_engineered_values() -> None:
    """Engineered features should be deterministic and easy to audit."""
    frame = pd.DataFrame(
        [
            {
                "amount": 100.0,
                "merchant_risk": 0.8,
                "customer_age": 35,
                "velocity_24h": 4,
            }
        ]
    )

    features = build_features(frame)

    assert features.loc[0, "amount_log"] == np.log1p(100.0)
    assert features.loc[0, "risk_velocity_interaction"] == 3.2
    assert features.loc[0, "amount_per_velocity"] == 20.0
    assert features.loc[0, "is_high_risk_merchant"] == 1


def test_amount_per_velocity_handles_zero_velocity() -> None:
    """Division by zero should not happen for first-time or quiet customers."""
    frame = pd.DataFrame(
        [
            {
                "amount": 250.0,
                "merchant_risk": 0.2,
                "customer_age": 50,
                "velocity_24h": 0,
            }
        ]
    )

    features = build_features(frame)

    assert features.loc[0, "amount_per_velocity"] == 250.0
