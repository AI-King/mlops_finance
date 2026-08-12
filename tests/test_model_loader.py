"""Tests for model loading configuration."""

from mlops_finance.model_loader import registry_model_uri


def test_registry_model_uri_uses_alias() -> None:
    """MLflow serving should target an alias, not a hard-coded version."""
    assert registry_model_uri("fraud-risk-model", "champion") == (
        "models:/fraud-risk-model@champion"
    )
