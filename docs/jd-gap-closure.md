# REIL JD gap closure

This project now demonstrates the main responsibilities in the REIL MLOps JD.

## Added capabilities

- `audit.py`: PostgreSQL audit records for model version, A/B variant, decision, probability, latency, fallback use, and eventual label.
- `rules.py`: deterministic fallback when model confidence is between 0.40 and 0.60.
- `retrain.py`: drift-triggered candidate retraining.
- `loadtest.py`: Locust test for latency and throughput measurement.
- `k8s/hpa.yaml`: automatic replica scaling.
- `terraform/main.tf`: infrastructure-as-code namespace starter.

## Commands

```powershell
$env:PYTHONPATH="src"
python -c "from mlops_finance.retrain import retrain_if_needed; print(retrain_if_needed('data/transactions.csv','data/feature_drift.csv'))"
locust -f loadtest.py --host http://localhost:8000
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/hpa.yaml
terraform -chdir=terraform init
terraform -chdir=terraform plan
```

## Important production decisions

The API intentionally continues serving when PostgreSQL is unavailable, returning `audit-unavailable`. In production, emit a Prometheus counter and alert on it; do not silently ignore audit loss. Use Alembic migrations, secrets from a vault, TLS, authentication, network policies, image scanning, immutable image tags, and an approval gate before production promotion.

The current A/B implementation is deterministic hashing. A production experiment should also persist assignment, outcome, cohort, sample size, confidence intervals, and a predeclared success metric.
