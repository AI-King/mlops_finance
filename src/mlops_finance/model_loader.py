"""Model loading helpers for local files and MLflow Registry aliases."""

from pathlib import Path
from typing import Any

import joblib
import mlflow.sklearn

from .config import settings

LOCAL_SOURCE = "local"
MLFLOW_SOURCE = "mlflow"


def registry_model_uri(model_name: str, alias: str) -> str:
    """Build an MLflow model URI for an alias such as champion or candidate.

    Complexity: O(1). DSA: string formatting only.
    """
    return f"models:/{model_name}@{alias}"


def load_model_from_settings() -> Any | None:
    """Load the configured champion model for API serving.

    Complexity: O(m), where m is model artifact size read from disk or MLflow.
    DSA: no custom data structure; the model object is deserialized from storage.
    """
    if settings.model_source == LOCAL_SOURCE:
        if not Path(settings.model_path).exists():
            return None
        return joblib.load(settings.model_path)

    if settings.model_source == MLFLOW_SOURCE:
        uri = registry_model_uri(
            settings.registered_model_name,
            settings.registered_model_alias,
        )
        return mlflow.sklearn.load_model(uri)

    raise ValueError(
        f"Unsupported model_source: {settings.model_source}. "
        f"Expected {LOCAL_SOURCE} or {MLFLOW_SOURCE}."
    )


def load_ab_models() -> dict[str, Any | None]:
    """Load model variants for deterministic A/B serving.

    Local mode supports a separate variant B file. MLflow mode serves the same
    champion model in both variants until variant-specific aliases are added.
    """
    model_a = load_model_from_settings()
    model_b = model_a
    if (
        settings.model_source == LOCAL_SOURCE
        and Path(settings.model_version_b_path).exists()
    ):
        model_b = joblib.load(settings.model_version_b_path)
    return {"a": model_a, "b": model_b}
