"""Drift-triggered retraining orchestration."""

import pandas as pd

from .monitoring import drift_report
from .train import train


def retrain_if_needed(reference_path: str, current_path: str,
                      threshold: float = 0.25) -> bool:
    """Retrain when any feature PSI crosses threshold.

    Complexity: O(f * n log b), where f is feature count. DSA: dictionary
    aggregation scans drift metrics and checks the maximum in O(f) time.
    """
    reference = pd.read_csv(reference_path)
    current = pd.read_csv(current_path)
    report = drift_report(reference, current)
    if max(report.values(), default=0.0) <= threshold:
        return False
    train(current_path, output="models/fraud_model_candidate.joblib")
    return True
