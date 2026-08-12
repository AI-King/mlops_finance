import pandas as pd

from mlops_finance.monitoring import psi


def test_psi_same_series_is_zero() -> None:
    values = pd.Series(range(100))
    assert psi(values, values) == 0.0
