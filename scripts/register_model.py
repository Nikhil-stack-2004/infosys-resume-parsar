"""Log the live model and metrics to an MLflow Model Registry."""

import os
import sys

import joblib
import mlflow
import mlflow.sklearn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scripts.evaluate_model import evaluate


model_path = os.path.join(ROOT, "models", "logistic_regression_model.pkl")
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:" + os.path.join(ROOT, "mlruns")))

with mlflow.start_run(run_name="career-area-logistic-regression") as run:
    metrics = evaluate()
    mlflow.log_metrics({"top1_accuracy": metrics["top1_accuracy"], "top3_accuracy": metrics["top3_accuracy"]})
    mlflow.log_param("career_area_classes", metrics["classes"])
    mlflow.sklearn.log_model(
        sk_model=joblib.load(model_path),
        artifact_path="career-predictor",
        registered_model_name="career-area-predictor",
    )
    print(f"registered run: {run.info.run_id}")
