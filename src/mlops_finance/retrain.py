"""Drift-triggered retraining orchestration."""

import mlflow
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
    max_drift = max(report.values(), default=0.0)
    if max_drift <= threshold:
        return False
    train(
        current_path,
        output=CANDIDATE_MODEL_PATH,
        model_role="candidate",
        run_reason="drift_retrain",
        extra_tags={"promotion_candidate": "true"},
    )
    candidate_metrics = load_metrics(CANDIDATE_MODEL_PATH)
    production_metrics = load_metrics(settings.model_path)
    if candidate_metrics is None:
        return False
    promoted = should_promote(candidate_metrics, production_metrics)
    with mlflow.start_run(run_name="promotion_decision"):
        mlflow.log_params(
            {
                "candidate_model_path": CANDIDATE_MODEL_PATH,
                "current_data_path": current_path,
                "max_drift": max_drift,
                "production_model_path": settings.model_path,
                "reference_data_path": reference_path,
                "threshold": threshold,
            }
        )
        mlflow.log_metrics(
            {
                "candidate_f1_score": candidate_metrics.f1_score,
                "candidate_precision": candidate_metrics.precision,
                "candidate_recall": candidate_metrics.recall,
                "candidate_roc_auc": candidate_metrics.roc_auc,
            }
        )
        mlflow.set_tags(
            {
                "model_role": "promotion_decision",
                "promoted": str(promoted).lower(),
                "run_reason": "drift_retrain",
            }
        )
    if promoted:
        promote_candidate(CANDIDATE_MODEL_PATH, settings.model_path)
        return True
    return False
