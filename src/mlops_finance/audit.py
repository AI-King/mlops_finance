"""PostgreSQL audit persistence for regulated model decisions."""

from datetime import datetime, timezone
from uuid import uuid4

import psycopg

from .config import settings


CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS prediction_audit (
    request_id UUID PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL,
    customer_id_hash TEXT NOT NULL,
    model_version TEXT NOT NULL,
    variant TEXT NOT NULL,
    probability DOUBLE PRECISION NOT NULL,
    decision TEXT NOT NULL,
    fallback_used BOOLEAN NOT NULL,
    latency_ms DOUBLE PRECISION NOT NULL,
    actual_label INTEGER
)
"""


def init_audit_table() -> None:
    """Create the audit table if PostgreSQL is available.

    Complexity: O(1) database operation. The production version should use
    migrations such as Alembic rather than creating tables at application boot.
    """
    with psycopg.connect(settings.database_url, connect_timeout=2) as connection:
        connection.execute(CREATE_TABLE_SQL)


def write_prediction(*, customer_hash: str, model_version: str, variant: str,
                     probability: float, decision: str, fallback_used: bool,
                     latency_ms: float) -> str:
    """Write one auditable prediction and return its UUID.

    DSA: UUID is a constant-time unique identifier; SQL indexing makes lookup
    by request ID approximately O(log n).
    """
    request_id = str(uuid4())
    query = """INSERT INTO prediction_audit
        (request_id, created_at, customer_id_hash, model_version, variant,
         probability, decision, fallback_used, latency_ms)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"""
    with psycopg.connect(settings.database_url, connect_timeout=2) as connection:
        connection.execute(query, (request_id, datetime.now(timezone.utc),
                                    customer_hash, model_version, variant,
                                    probability, decision, fallback_used,
                                    latency_ms))
    return request_id
