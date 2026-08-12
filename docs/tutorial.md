# Tutorial

```bash
python -m mlops_finance.cli generate-data
python -m mlops_finance.cli train
python -m mlops_finance.cli generate-data --drift feature_drift
python -m mlops_finance.cli generate-data --drift concept_drift
pytest
```

For a real pipeline, retain separate train/validation/test periods to prevent temporal leakage. Register models only when quality, bias, security, and data-contract checks pass. Deploy a candidate to variant B, compare business and technical metrics, then promote or roll back. For CI/CD, build an immutable image tagged with the git SHA, scan it, push it, run Kubernetes smoke tests, and use a manual approval for production.

## Model registry flow

This project serves the MLflow registered champion model by default and keeps a
local model file as a fallback for simple Docker or offline development.

- `models/fraud_model.joblib`: local production/champion fallback model.
- `models/fraud_model_candidate.joblib`: local candidate/challenger model trained
  after drift.
- `fraud-risk-model@champion`: MLflow alias for the model version considered
  production-ready.
- `fraud-risk-model@candidate`: MLflow alias for the latest retrained candidate.

Training logs a model version to MLflow. Retraining marks it as `candidate`.
If promotion gates pass, the local candidate replaces the production fallback
file and the MLflow `champion` alias is moved to the candidate version. Set
`MODEL_SOURCE=local` only when MLflow registry/artifacts are unavailable.

DSA appears in vectorized arrays, histogram buckets for PSI, tree nodes in the model, hash buckets for stable A/B assignment, and dictionary lookups for in-memory model routing. Complexity comments in each function make the dominant cost explicit.
