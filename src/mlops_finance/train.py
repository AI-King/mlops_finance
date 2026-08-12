"""Training and MLflow tracking."""
from pathlib import Path
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from .config import settings
from .data import FEATURES


TEST_SIZE = 0.2
RANDOM_STATE = 42
MAX_ITER = 120


def train(path: str = "data/transactions.csv", output: str | None = None) -> float:
    """Train, evaluate, track, and save a classifier.

    Complexity: roughly O(n * t * log n), driven by tree training. DSA: tree nodes.
    """
    target = output or settings.model_path

    # Read the prepared training dataset. Complexity: O(n), one pass over rows.
    frame = pd.read_csv(path)

    # Split features (X) from label (y). DSA: DataFrame column selection uses indexed
    # column lookup, then creates tabular arrays for model training.
    x_train, x_test, y_train, y_test = train_test_split(
        frame[FEATURES],
        frame["fraud"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=frame["fraud"],
    )

    # Gradient boosting builds many decision trees. DSA: each tree repeatedly searches
    # feature split points to reduce classification error.
    model = HistGradientBoostingClassifier(
        max_iter=MAX_ITER,
        random_state=RANDOM_STATE,
    )

    with mlflow.start_run():
        model.fit(x_train, y_train)
        fraud_probabilities = model.predict_proba(x_test)[:, 1]
        predictions = (fraud_probabilities >= 0.5).astype(int)
        auc = roc_auc_score(y_test, fraud_probabilities)
        precision = precision_score(y_test, predictions, zero_division=0)
        recall = recall_score(y_test, predictions, zero_division=0)
        f1 = f1_score(y_test, predictions, zero_division=0)
        matrix = confusion_matrix(y_test, predictions)

        # Parameters explain how this run was produced. Metrics explain how well it did.
        mlflow.log_params({
            "dataset_path": path,
            "features": ",".join(FEATURES),
            "label": "fraud",
            "model_type": type(model).__name__,
            "max_iter": MAX_ITER,
            "random_state": RANDOM_STATE,
            "test_size": TEST_SIZE,
            "train_rows": len(x_train),
            "test_rows": len(x_test),
            "total_rows": len(frame),
            "model_output_path": target,
        })
        mlflow.log_metrics({
            "roc_auc": auc,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "true_negatives": int(matrix[0][0]),
            "false_positives": int(matrix[0][1]),
            "false_negatives": int(matrix[1][0]),
            "true_positives": int(matrix[1][1]),
        })

        # Confusion matrix is also saved as a readable table for audits and reports.
        confusion_frame = pd.DataFrame(
            matrix,
            index=["actual_not_fraud", "actual_fraud"],
            columns=["predicted_not_fraud", "predicted_fraud"],
        )
        mlflow.log_table(confusion_frame, artifact_file="confusion_matrix.json")

        # Store the model inside MLflow so runs can be compared and promoted later.
        mlflow.sklearn.log_model(model, name="model")

    Path(target).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, target)
    return float(auc)
