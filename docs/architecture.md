# Architecture and concepts

`data.py` generates reproducible synthetic transactions. `train.py` creates a time-stamped MLflow run, logs metrics/artifacts, and writes a serving artifact. The API loads artifacts once at startup, validates requests, assigns customers to a stable A/B bucket, and exports request count/latency to Prometheus. Grafana visualizes latency, traffic, errors, and variant outcomes. PostgreSQL is the natural audit store for request ID, model version, input hash, prediction, and outcome; keep raw sensitive data out of logs.

Feature drift means input distributions change while the label relationship is unchanged; this project simulates it by scaling amount and velocity. Concept drift means the relationship between features and label changes; this project changes the scoring equation. PSI is a simple first detector: commonly, <0.1 is low, 0.1–0.25 warrants investigation, and >0.25 is significant. Thresholds must be calibrated to your domain.

Production reliability: use timeouts, bounded payloads, retries only for safe operations, idempotency keys, readiness/liveness probes, multiple replicas, resource limits, rolling deployments, canary/A-B analysis, alerting, and rollback to the last known-good MLflow model. Model quality alerts need delayed ground truth; operational alerts can be immediate.

Power BI should read a curated PostgreSQL view or exported monitoring table, not the online prediction path. Recommended dimensions: date, model version, variant, segment, prediction bucket, latency bucket, and drift status.
