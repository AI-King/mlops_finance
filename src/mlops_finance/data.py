"""Synthetic financial transactions and controlled drift."""
from pathlib import Path
import numpy as np
import pandas as pd

FEATURES = ["amount", "merchant_risk", "customer_age", "velocity_24h"]


def make_data(rows: int = 10_000, seed: int = 42, drift: str = "normal") -> pd.DataFrame:
    """Create labels with optional feature or concept drift.

    Complexity: O(n) time/space. DSA: vectorized NumPy arrays avoid a Python loop.
    """
    rng = np.random.default_rng(seed)
    amount = rng.lognormal(3.5, 1.0, rows)
    risk = rng.beta(2, 8, rows)
    age = rng.normal(42, 12, rows).clip(18, 90)
    velocity = rng.poisson(3, rows)
    if drift == "feature_drift":
        amount *= 1.8
        velocity += 3
    score = 0.004 * amount + 3 * risk + 0.15 * velocity - 1.7
    if drift == "concept_drift":
        score = 0.002 * amount + 5 * risk + 0.35 * velocity - 2.4
    probability = 1 / (1 + np.exp(-score))
    return pd.DataFrame({"amount": amount, "merchant_risk": risk,
                         "customer_age": age, "velocity_24h": velocity,
                         "fraud": rng.binomial(1, probability)})


def save_data(frame: pd.DataFrame, path: str = "data/transactions.csv") -> None:
    """Save data and create its parent directory."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
