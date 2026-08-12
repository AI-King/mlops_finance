"""Tests for experiment metadata helpers."""

from mlops_finance.metadata import current_git_sha


def test_current_git_sha_returns_string() -> None:
    """Git SHA helper should never break experiment logging."""
    assert isinstance(current_git_sha(), str)
    assert current_git_sha()
