"""Drift-triggered retraining orchestration."""

import pandas as pd

from .config import settings
from .monitoring import drift_report
from .registry import load_metrics, promote_candidate, should_promote
from .train import train

CANDIDATE_MODEL_PATH = "models/fraud_model_candidate.joblib"


def retrain_if_needed(
    reference_path: str, current_path: str, threshold: float = 0.25
) -> bool:
    """Retrain on drift and promote only when candidate quality is acceptable.

    Complexity: O(f * n log b), where f is feature count. DSA: dictionary
    aggregation scans drift metrics and checks the maximum in O(f) time.
    """
    reference = pd.read_csv(reference_path)
    current = pd.read_csv(current_path)
    report = drift_report(reference, current)
    if max(report.values(), default=0.0) <= threshold:
        return False
    train(current_path, output=CANDIDATE_MODEL_PATH)
    candidate_metrics = load_metrics(CANDIDATE_MODEL_PATH)
    production_metrics = load_metrics(settings.model_path)
    if candidate_metrics is None:
        return False
    if should_promote(candidate_metrics, production_metrics):
        promote_candidate(CANDIDATE_MODEL_PATH, settings.model_path)
        return True
    return False
