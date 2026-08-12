"""Tests for MLflow registry helper logic."""

from mlops_finance.mlflow_registry import (
    CANDIDATE_ALIAS,
    CHAMPION_ALIAS,
    alias_for_role,
)


def test_alias_for_role_maps_production_to_champion() -> None:
    """Production is the local name for the currently served champion model."""
    assert alias_for_role("production") == CHAMPION_ALIAS


def test_alias_for_role_maps_candidate_to_candidate() -> None:
    """Candidate models should be easy to find in the MLflow registry."""
    assert alias_for_role("candidate") == CANDIDATE_ALIAS


def test_alias_for_role_ignores_non_model_roles() -> None:
    """Promotion decision runs are not model versions and should not get aliases."""
    assert alias_for_role("promotion_decision") is None
