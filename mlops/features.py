from pathlib import Path
import typer
from loguru import logger
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
from mlops.config import PROCESSED_DATA_DIR, MODELS_DIR

app = typer.Typer()

COLS_TO_SCALE = ['Time', 'Amount']


def preprocess(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    if train_df.empty or test_df.empty:
        raise ValueError("Input DataFrames cannot be empty")

    for col in COLS_TO_SCALE:
        if col in train_df.columns and train_df[col].isnull().any():
            raise ValueError(f"Column '{col}' in train_df contains NaN values")
        if col in test_df.columns and test_df[col].isnull().any():
            raise ValueError(f"Column '{col}' in test_df contains NaN values")

    scaler = StandardScaler()
    train_scaled = train_df.copy()
    test_scaled = test_df.copy()

    train_scaled[COLS_TO_SCALE] = scaler.fit_transform(train_df[COLS_TO_SCALE])
    test_scaled[COLS_TO_SCALE] = scaler.transform(test_df[COLS_TO_SCALE])

    return train_scaled, test_scaled, scaler


@app.command()
def main(
    input_path: Path = typer.Option(PROCESSED_DATA_DIR),
    output_path: Path = typer.Option(PROCESSED_DATA_DIR),
    model_dir: Path = typer.Option(MODELS_DIR),
):
    logger.info("Loading data...")
    try:
        train_df = pd.read_csv(input_path / "train.csv")
        test_df = pd.read_csv(input_path / "test.csv")
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise typer.Exit(code=1)

    logger.info("Preprocessing features...")
    train_scaled, test_scaled, scaler = preprocess(train_df, test_df)

    output_path.mkdir(parents=True, exist_ok=True)
    train_output = output_path / "train_featured.csv"
    test_output = output_path / "test_featured.csv"

    train_scaled.to_csv(train_output, index=False)
    test_scaled.to_csv(test_output, index=False)

    model_dir.mkdir(parents=True, exist_ok=True)
    scaler_path = model_dir / "scaler.pkl"
    joblib.dump(scaler, scaler_path)

    logger.success(f"Processed train data saved to {train_output}")
    logger.success(f"Processed test data saved to {test_output}")
    logger.success(f"Scaler saved to {scaler_path}")


if __name__ == "__main__":
    app()
