import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

FEATURE_COLS = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
ALL_COLS = FEATURE_COLS + ['Class']
N_FEATURES = len(FEATURE_COLS)  # 30


def make_transactions_df(n: int = 50, random_seed: int = 42, class_val: int | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)
    data = {col: rng.standard_normal(n) for col in FEATURE_COLS}
    data['Class'] = np.full(n, class_val, dtype=int) if class_val is not None else rng.integers(0, 2, n)
    return pd.DataFrame(data)


@pytest.fixture
def normal_train_df() -> pd.DataFrame:
    return make_transactions_df(n=100)


@pytest.fixture
def normal_test_df() -> pd.DataFrame:
    return make_transactions_df(n=20, random_seed=99)


@pytest.fixture(scope="session")
def mock_rf_model() -> RandomForestClassifier:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((120, N_FEATURES))
    y = np.array([0] * 100 + [1] * 20)
    clf = RandomForestClassifier(n_estimators=5, random_state=0)
    clf.fit(X, y)
    return clf


@pytest.fixture(scope="session")
def mock_scaler() -> StandardScaler:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((120, 2))
    scaler = StandardScaler()
    scaler.fit(X)
    return scaler


@pytest.fixture
def client(monkeypatch, mock_rf_model, mock_scaler) -> TestClient:
    import mlops.modeling.predict as predict_module

    monkeypatch.setattr(predict_module, "model", mock_rf_model)
    monkeypatch.setattr(predict_module, "scaler", mock_scaler)
    return TestClient(predict_module.app)
