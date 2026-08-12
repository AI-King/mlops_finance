"""Tests for local model promotion policy."""

from pathlib import Path

from mlops_finance.registry import (
    ModelMetrics,
    load_metrics,
    promote_candidate,
    save_metrics,
    should_promote,
)


def test_should_promote_when_no_production_metrics_exist() -> None:
    """The first valid candidate can become production."""
    candidate = ModelMetrics(
        roc_auc=0.7,
        precision=0.6,
        recall=0.5,
        f1_score=0.55,
    )

    assert should_promote(candidate, production=None)


def test_should_promote_rejects_lower_recall() -> None:
    """Fraud models should not silently miss more fraud after promotion."""
    production = ModelMetrics(
        roc_auc=0.7,
        precision=0.6,
        recall=0.5,
        f1_score=0.55,
    )
    candidate = ModelMetrics(
        roc_auc=0.72,
        precision=0.7,
        recall=0.4,
        f1_score=0.56,
    )

    assert not should_promote(candidate, production)


def test_should_promote_allows_small_auc_drop_when_recall_and_f1_hold() -> None:
    """A tiny ROC-AUC drop is acceptable when fraud-oriented metrics are stable."""
    production = ModelMetrics(
        roc_auc=0.70,
        precision=0.60,
        recall=0.50,
        f1_score=0.55,
    )
    candidate = ModelMetrics(
        roc_auc=0.69,
        precision=0.61,
        recall=0.52,
        f1_score=0.56,
    )

    assert should_promote(candidate, production)


def test_save_and_load_metrics_round_trip(tmp_path: Path) -> None:
    """Metrics sidecar JSON should preserve promotion inputs."""
    model_path = tmp_path / "fraud_model.joblib"
    metrics = ModelMetrics(
        roc_auc=0.7,
        precision=0.6,
        recall=0.5,
        f1_score=0.55,
    )

    save_metrics(str(model_path), metrics)

    assert load_metrics(str(model_path)) == metrics


def test_promote_candidate_copies_model_and_metrics(tmp_path: Path) -> None:
    """Promotion should update both model artifact and metric evidence."""
    candidate_path = tmp_path / "candidate.joblib"
    production_path = tmp_path / "production.joblib"
    candidate_path.write_text("candidate-model", encoding="utf-8")
    metrics = ModelMetrics(
        roc_auc=0.7,
        precision=0.6,
        recall=0.5,
        f1_score=0.55,
    )
    save_metrics(str(candidate_path), metrics)

    promote_candidate(str(candidate_path), str(production_path))

    assert production_path.read_text(encoding="utf-8") == "candidate-model"
    assert load_metrics(str(production_path)) == metrics
