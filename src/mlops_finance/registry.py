"""Local model registry and promotion policy.

This is a learning-friendly stand-in for a managed MLflow Model Registry flow.
"""

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelMetrics:
    """Quality metrics used to decide whether a model can be promoted."""

    roc_auc: float
    precision: float
    recall: float
    f1_score: float


def metrics_path(model_path: str) -> Path:
    """Return the sidecar metrics JSON path for a model artifact.

    Complexity: O(1). DSA: path manipulation is constant-time string handling.
    """
    return Path(f"{model_path}.metrics.json")


def save_metrics(model_path: str, metrics: ModelMetrics) -> None:
    """Persist model quality metrics beside the model file."""
    path = metrics_path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(metrics), indent=2), encoding="utf-8")


def load_metrics(model_path: str) -> ModelMetrics | None:
    """Load model metrics if they exist."""
    path = metrics_path(model_path)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return ModelMetrics(**data)


def should_promote(
    candidate: ModelMetrics,
    production: ModelMetrics | None,
    max_roc_auc_drop: float = 0.02,
) -> bool:
    """Return whether a candidate model is good enough for production.

    Complexity: O(1), fixed number of metric comparisons.
    DSA: boolean conditions combine promotion gates like a decision table.
    """
    if production is None:
        return True

    recall_not_worse = candidate.recall >= production.recall
    f1_not_worse = candidate.f1_score >= production.f1_score
    auc_not_too_low = candidate.roc_auc >= production.roc_auc - max_roc_auc_drop
    return recall_not_worse and f1_not_worse and auc_not_too_low


def promote_candidate(candidate_path: str, production_path: str) -> None:
    """Copy candidate model and metrics into the production model location."""
    Path(production_path).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(candidate_path, production_path)

    candidate_metrics_path = metrics_path(candidate_path)
    if candidate_metrics_path.exists():
        shutil.copy2(candidate_metrics_path, metrics_path(production_path))
