"""MLflow Model Registry helpers."""

import logging
from typing import Any

from mlflow.tracking import MlflowClient

REGISTERED_MODEL_NAME = "fraud-risk-model"
CHAMPION_ALIAS = "champion"
CANDIDATE_ALIAS = "candidate"
LOGGER = logging.getLogger(__name__)


def alias_for_role(model_role: str) -> str | None:
    """Return the MLflow registry alias for a training role.

    Complexity: O(1). DSA: dictionary lookup maps local role names to aliases.
    """
    aliases = {
        "production": CHAMPION_ALIAS,
        "candidate": CANDIDATE_ALIAS,
    }
    return aliases.get(model_role)


def set_model_alias(
    model_version: Any | None,
    alias: str | None,
    model_name: str = REGISTERED_MODEL_NAME,
) -> None:
    """Point an MLflow registered model alias at a model version if available."""
    if model_version is None or alias is None:
        return
    version = getattr(model_version, "version", model_version)
    try:
        MlflowClient().set_registered_model_alias(
            name=model_name,
            alias=alias,
            version=str(version),
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Could not set MLflow alias %s: %s", alias, exc)


def promote_candidate_alias(model_name: str = REGISTERED_MODEL_NAME) -> None:
    """Move the champion alias to the version currently marked candidate."""
    try:
        candidate = MlflowClient().get_model_version_by_alias(
            name=model_name,
            alias=CANDIDATE_ALIAS,
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Could not read MLflow candidate alias: %s", exc)
        return
    set_model_alias(candidate, CHAMPION_ALIAS, model_name=model_name)
