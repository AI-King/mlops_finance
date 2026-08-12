"""FastAPI serving with health checks, metrics, and deterministic A/B tests."""

import hashlib
import logging
import time
from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Histogram, make_asgi_app
from pydantic import BaseModel, Field

from .config import settings
from .features import build_features
from .rules import rule_based_decision

REQUESTS = Counter("prediction_requests_total", "Prediction requests", ["variant"])
LATENCY = Histogram("prediction_latency_seconds", "Prediction latency")
LOGGER = logging.getLogger(__name__)
models: dict[str, object] = {}


class Transaction(BaseModel):
    """Validated transaction payload."""

    amount: float = Field(gt=0)
    merchant_risk: float = Field(ge=0, le=1)
    customer_age: float = Field(ge=18, le=100)
    velocity_24h: int = Field(ge=0)
    customer_id: str = "anonymous"


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Load artifacts once per process; O(1) dictionary setup."""
    try:
        models["a"] = joblib.load(settings.model_path)
        try:
            models["b"] = joblib.load(settings.model_version_b_path)
        except FileNotFoundError:
            models["b"] = models["a"]
    except FileNotFoundError:
        models["a"] = None
    # Initialize audit storage once, not for every request. If PostgreSQL is
    # unavailable, readiness still depends only on the model and requests can
    # continue with an explicit audit-unavailable marker.
    try:
        from .audit import init_audit_table

        init_audit_table()
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Audit table initialization failed: %s", exc)
    yield


app = FastAPI(title="Financial Fraud Model", lifespan=lifespan)
app.mount("/metrics", make_asgi_app())


@app.get("/health/live")
def live() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "alive"}


@app.get("/health/ready")
def ready() -> dict[str, str]:
    """Readiness probe requiring a loaded model."""
    if models.get("a") is None:
        raise HTTPException(status_code=503, detail="model unavailable")
    return {"status": "ready"}


@app.post("/predict")
def predict(transaction: Transaction) -> dict[str, float | str]:
    """Return fraud probability and A/B variant.

    Complexity: O(1) routing plus model inference. DSA: hash bucket assignment.
    """
    started = time.perf_counter()
    with LATENCY.time():
        bucket = (
            int(hashlib.md5(transaction.customer_id.encode()).hexdigest(), 16) % 100
        )
        variant = "b" if bucket < settings.ab_test_percent else "a"
        model = models.get(variant) or models.get("a")
        if model is None:
            raise HTTPException(status_code=503, detail="model unavailable")
        raw_features = pd.DataFrame([transaction.model_dump()])
        features = build_features(raw_features)
        probability = float(model.predict_proba(features)[0][1])
        # A probability near 0.5 means the model is uncertain. In that case,
        # use deterministic business rules instead of making an unreliable call.
        fallback_used = settings.fallback_low < probability < settings.fallback_high
        if fallback_used:
            decision, reason = rule_based_decision(
                transaction.amount,
                transaction.merchant_risk,
                transaction.velocity_24h,
            )
        else:
            decision = "review" if probability >= 0.5 else "approve"
            reason = "model_high_confidence"
        REQUESTS.labels(variant).inc()
        try:
            from .audit import write_prediction

            request_id = write_prediction(
                customer_hash=hashlib.sha256(
                    transaction.customer_id.encode()
                ).hexdigest(),
                model_version=settings.model_version,
                variant=variant,
                probability=probability,
                decision=decision,
                fallback_used=fallback_used,
                latency_ms=(time.perf_counter() - started) * 1000,
            )
        except Exception as exc:  # noqa: BLE001
            # Availability first: an audit database outage must not take down
            # online inference. Alert on this exception in production.
            LOGGER.warning("Audit write failed: %s", exc)
            request_id = "audit-unavailable"
        return {
            "request_id": request_id,
            "fraud_probability": probability,
            "variant": variant,
            "decision": decision,
            "reason": reason,
            "fallback_used": fallback_used,
        }
