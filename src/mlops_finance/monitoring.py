"""Explainable feature-drift monitoring."""

import numpy as np
import pandas as pd

from .data import FEATURES


def psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """Calculate population stability index.

    Complexity: O(n log b); DSA: histogram frequency arrays indexed by buckets.
    """
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0
    expected = np.histogram(reference, edges)[0] / len(reference)
    actual = np.histogram(current, edges)[0] / len(current)
    expected = np.clip(expected, 1e-6, None)
    actual = np.clip(actual, 1e-6, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def drift_report(reference: pd.DataFrame, current: pd.DataFrame) -> dict[str, float]:
    """Return PSI per model feature."""
    return {feature: psi(reference[feature], current[feature]) for feature in FEATURES}
