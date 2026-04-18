from pathlib import Path

import arff
import pandas as pd
import typer
from loguru import logger
from sklearn.model_selection import train_test_split

from mlops.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    input_path: Path = typer.Option(RAW_DATA_DIR / "dataset"),
    output_path: Path = typer.Option(PROCESSED_DATA_DIR),
    test_size: float = typer.Option(0.2),
    random_state: int = typer.Option(42),
):
    logger.info("Loading dataset...")

    with open(input_path) as f:
        dataset = arff.load(f)

    col_names = [attr[0] for attr in dataset['attributes']]
    df = pd.DataFrame(dataset['data'], columns=col_names)

    logger.info(f"Dataset shape: {df.shape}")

    logger.info("Splitting data into train and test sets...")
    train, test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df['Class']
    )

    output_path.mkdir(parents=True, exist_ok=True)
    train_path = output_path / "train.csv"
    test_path = output_path / "test.csv"

    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)

    logger.success(f"Train data saved to {train_path}")
    logger.success(f"Test data saved to {test_path}")


if __name__ == "__main__":
    app()
