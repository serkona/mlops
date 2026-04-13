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
    if class_val is not None:
        data['Class'] = np.full(n, class_val, dtype=int)
    else:
        data['Class'] = rng.integers(0, 2, n)
    return pd.DataFrame(data)


@pytest.fixture
def normal_train_df() -> pd.DataFrame:
    return make_transactions_df(n=100)


@pytest.fixture
def normal_test_df() -> pd.DataFrame:
    return make_transactions_df(n=20, random_seed=99)


VALID_TRANSACTION = {
    "Time": 0.0,
    "V1": -1.3598071336738,
    "V2": -0.0727811733098497,
    "V3": 2.53634673796914,
    "V4": 1.37815522427443,
    "V5": -0.338320769942518,
    "V6": 0.462387777762292,
    "V7": 0.239598554061257,
    "V8": 0.0986979012610507,
    "V9": 0.363786969611213,
    "V10": 0.0907941719789316,
    "V11": -0.551599533260813,
    "V12": -0.617800855762348,
    "V13": -0.991389847235408,
    "V14": -0.311169353699879,
    "V15": 1.46817697209427,
    "V16": -0.470400525259478,
    "V17": 0.207971241929242,
    "V18": 0.0257905801985591,
    "V19": 0.403992960255733,
    "V20": 0.251412098239705,
    "V21": -0.018306777944153,
    "V22": 0.277837575558899,
    "V23": -0.110473910188767,
    "V24": 0.0669280749146731,
    "V25": 0.128539358273528,
    "V26": -0.189114843888824,
    "V27": 0.133558376740387,
    "V28": -0.0210530534538215,
    "Amount": 149.62,
}


@pytest.fixture(scope="session")
def valid_transaction() -> dict:
    return VALID_TRANSACTION.copy()


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
