"""Shared feature engineering for training and online inference."""

import numpy as np
import pandas as pd

RAW_FEATURES = ["amount", "merchant_risk", "customer_age", "velocity_24h"]
MODEL_FEATURES = [
    *RAW_FEATURES,
    "amount_log",
    "risk_velocity_interaction",
    "amount_per_velocity",
    "is_high_risk_merchant",
]


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create model-ready features from validated transaction data.

    Complexity: O(n), one vectorized pass per engineered column.
    DSA: pandas Series are column arrays, so operations run over contiguous vectors.
    """
    features = frame[RAW_FEATURES].copy()

    # Log transform reduces the impact of very large transaction amounts.
    features["amount_log"] = np.log1p(features["amount"])

    # Interaction captures the combined risk of risky merchants and repeated activity.
    features["risk_velocity_interaction"] = (
        features["merchant_risk"] * features["velocity_24h"]
    )

    # Add 1 to avoid division by zero for customers with no recent transactions.
    features["amount_per_velocity"] = features["amount"] / (
        features["velocity_24h"] + 1
    )

    # Business-rule style binary feature: high risk merchant or not.
    features["is_high_risk_merchant"] = (features["merchant_risk"] >= 0.7).astype(int)

    return features[MODEL_FEATURES]
