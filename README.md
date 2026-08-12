# Financial MLOps Reference Project

End-to-end synthetic fraud detection with modular Python, MLflow, Docker, Kubernetes, PostgreSQL-ready auditing, CI/CD, serving, Prometheus/Grafana, feature drift, concept drift, and A/B testing.

## Quick start

```bash
.venv\\Scripts\\activate
```

The repository already contains the local `.venv`. To recreate it from scratch:

```powershell
python -m venv .venv
.\.venv\\Scripts\\python.exe -m pip install -r requirements.txt
```

Then run:

```bash
python -m mlops_finance.cli generate-data
python -m mlops_finance.cli train
uvicorn mlops_finance.api:app --reload
```

By default, the API serves the MLflow registered champion model:

```text
models:/fraud-risk-model@champion
```

Use local-file serving only when you explicitly want it:

```powershell
$env:MODEL_SOURCE="local"
uvicorn mlops_finance.api:app --reload
```

Run `docker compose up --build` for PostgreSQL, MLflow, API, Prometheus, and Grafana. API: `localhost:8000`; MLflow: `localhost:5000`; Grafana: `localhost:3000` (`admin/admin`).

Docker Compose uses its own persisted MLflow store in `mlflow-docker/`. After
starting Compose for the first time, register a champion model in that store:

```powershell
$env:MLFLOW_TRACKING_URI="http://localhost:5000"
$env:PYTHONPATH="src"
python -m mlops_finance.cli train
```

Then restart the API container so it loads `fraud-risk-model@champion` from the
Compose MLflow service.

For live dashboard traffic, keep Compose running and start the generator:

```powershell
python traffic_generator.py --mode normal --interval 1
```

Open another terminal to simulate feature drift:

```powershell
python traffic_generator.py --mode feature_drift --interval 1
```

Stop either generator with `Ctrl+C`. The generator continuously updates API
metrics, PostgreSQL audit rows, and Prometheus/Grafana dashboards.

Read [docs/architecture.md](docs/architecture.md) and [docs/tutorial.md](docs/tutorial.md).

This is a teaching reference. Add secrets management, encryption, IAM, audit retention, explainability, fairness, approval workflows, and regulatory controls before production.
