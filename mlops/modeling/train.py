import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import typer
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from mlops.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    train_path: Path = PROCESSED_DATA_DIR / "train_featured.csv",
    test_path: Path = PROCESSED_DATA_DIR / "test_featured.csv",
    model_dir: Path = MODELS_DIR,
    n_estimators: int = 100,
    max_depth: int = None,
    random_state: int = 42,
):
    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    logger.info(f"MLflow tracking URI: {mlflow_tracking_uri}")

    mlflow.set_experiment("credit-card-fraud")

    with mlflow.start_run():
        logger.info("Loading data...")
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
            raise typer.Exit(code=1)

        X_train = train_df.drop("Class", axis=1)
        y_train = train_df["Class"]
        X_test = test_df.drop("Class", axis=1)
        y_test = test_df["Class"]

        logger.info("Training Random Forest model...")
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )
        model.fit(X_train, y_train)

        logger.info("Evaluating model...")
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        logger.info(f"Accuracy:  {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall:    {recall:.4f}")
        logger.info(f"F1 Score:  {f1:.4f}")

        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", random_state)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)

        mlflow.sklearn.log_model(model, "model")

        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "model.pkl"
        joblib.dump(model, model_path)
        logger.success(f"Model saved to {model_path}")


if __name__ == "__main__":
    app()
