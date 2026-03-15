from pathlib import Path
import typer
from loguru import logger
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
from mlops.config import PROCESSED_DATA_DIR, MODELS_DIR

app = typer.Typer()

@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR,
    output_path: Path = PROCESSED_DATA_DIR,
    model_dir: Path = MODELS_DIR,
):
    logger.info("Loading data...")
    try:
        train_df = pd.read_csv(input_path / "train.csv")
        test_df = pd.read_csv(input_path / "test.csv")
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise typer.Exit(code=1)

    logger.info("Preprocessing features...")
    scaler = StandardScaler()
    cols_to_scale = ['Time', 'Amount']
    
    train_df[cols_to_scale] = scaler.fit_transform(train_df[cols_to_scale])
    test_df[cols_to_scale] = scaler.transform(test_df[cols_to_scale])

    output_path.mkdir(parents=True, exist_ok=True)
    train_output = output_path / "train_featured.csv"
    test_output = output_path / "test_featured.csv"
    
    train_df.to_csv(train_output, index=False)
    test_df.to_csv(test_output, index=False)
    
    model_dir.mkdir(parents=True, exist_ok=True)
    scaler_path = model_dir / "scaler.pkl"
    joblib.dump(scaler, scaler_path)
    
    logger.success(f"Processed train data saved to {train_output}")
    logger.success(f"Processed test data saved to {test_output}")
    logger.success(f"Scaler saved to {scaler_path}")

if __name__ == "__main__":
    app()
