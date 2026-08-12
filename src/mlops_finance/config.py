"""Environment-backed application settings."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Runtime settings. Complexity and DSA: O(1) key/value lookup."""

    model_source: str = "mlflow"
    model_path: str = "models/fraud_model.joblib"
    model_version_b_path: str = "models/fraud_model_b.joblib"
    registered_model_name: str = "fraud-risk-model"
    registered_model_alias: str = "champion"
    ab_test_percent: int = 50
    database_url: str = "postgresql://mlops:mlops@localhost:5432/mlops"
    model_version: str = "fraud-model-v1"
    fallback_low: float = 0.40
    fallback_high: float = 0.60


settings = Settings()
