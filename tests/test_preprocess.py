import numpy as np
import pandas as pd
import pytest
import typer

from mlops.features import COLS_TO_SCALE, main as features_main, preprocess

FEATURE_COLS = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']


def _make_df(n: int = 50, random_seed: int = 42, class_val: int | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)
    data = {col: rng.standard_normal(n) for col in FEATURE_COLS}
    data['Class'] = np.full(n, class_val, dtype=int) if class_val is not None else rng.integers(0, 2, n)
    return pd.DataFrame(data)


def _nan_amount_df(n: int = 50) -> pd.DataFrame:
    df = _make_df(n)
    df['Amount'] = float('nan')
    return df


def _string_amount_df(n: int = 50) -> pd.DataFrame:
    df = _make_df(n)
    df['Amount'] = 'not_a_number'
    return df


class TestPreprocessNormalBehaviour:
    def test_output_shapes_match_input(self, normal_train_df, normal_test_df):
        train_out, test_out, _ = preprocess(normal_train_df, normal_test_df)
        assert train_out.shape == normal_train_df.shape
        assert test_out.shape == normal_test_df.shape

    def test_column_names_preserved(self, normal_train_df, normal_test_df):
        train_out, test_out, _ = preprocess(normal_train_df, normal_test_df)
        assert list(train_out.columns) == list(normal_train_df.columns)
        assert list(test_out.columns) == list(normal_test_df.columns)

    def test_time_and_amount_are_scaled(self, normal_train_df, normal_test_df):
        train_out, _, _ = preprocess(normal_train_df, normal_test_df)
        # After fit_transform on training data the mean should be ~0 and std ~1
        for col in COLS_TO_SCALE:
            assert abs(train_out[col].mean()) < 0.1, f"{col} mean should be near 0"
            assert abs(train_out[col].std() - 1.0) < 0.1, f"{col} std should be near 1"

    def test_v_columns_unchanged(self, normal_train_df, normal_test_df):
        train_out, _, _ = preprocess(normal_train_df, normal_test_df)
        v_cols = [f'V{i}' for i in range(1, 29)]
        for col in v_cols:
            pd.testing.assert_series_equal(
                train_out[col], normal_train_df[col], check_names=True
            )

    def test_class_column_unchanged(self, normal_train_df, normal_test_df):
        train_out, test_out, _ = preprocess(normal_train_df, normal_test_df)
        pd.testing.assert_series_equal(train_out['Class'], normal_train_df['Class'])
        pd.testing.assert_series_equal(test_out['Class'], normal_test_df['Class'])

    def test_scaler_fitted_on_train_size(self, normal_train_df, normal_test_df):
        _, _, scaler = preprocess(normal_train_df, normal_test_df)
        assert scaler.n_samples_seen_ == len(normal_train_df)

    def test_returns_standard_scaler(self, normal_train_df, normal_test_df):
        from sklearn.preprocessing import StandardScaler
        _, _, scaler = preprocess(normal_train_df, normal_test_df)
        assert isinstance(scaler, StandardScaler)

    def test_original_dataframes_not_mutated(self, normal_train_df, normal_test_df):
        original_train_time = normal_train_df['Time'].copy()
        original_test_amount = normal_test_df['Amount'].copy()
        preprocess(normal_train_df, normal_test_df)
        pd.testing.assert_series_equal(normal_train_df['Time'], original_train_time)
        pd.testing.assert_series_equal(normal_test_df['Amount'], original_test_amount)



class TestPreprocessSingleClass:
    @pytest.mark.parametrize("class_val", [0, 1])
    def test_single_class_succeeds(self, class_val):
        train_df = _make_df(n=100, class_val=class_val)
        test_df = _make_df(n=20, class_val=class_val)
        train_out, test_out, _ = preprocess(train_df, test_df)
        assert train_out.shape == train_df.shape
        assert test_out.shape == test_df.shape

    def test_single_class_labels_unchanged(self):
        train_df = _make_df(n=60, class_val=1)
        test_df = _make_df(n=15, class_val=1)
        train_out, test_out, _ = preprocess(train_df, test_df)
        assert (train_out['Class'] == 1).all()
        assert (test_out['Class'] == 1).all()



@pytest.mark.parametrize(
    "get_train, get_test, description",
    [
        (
            lambda: pd.DataFrame(),
            lambda: pd.DataFrame(),
            "empty_dataframes",
        ),
        (
            lambda: pd.DataFrame(),
            lambda: _make_df(20),
            "empty_train_only",
        ),
        (
            lambda: _make_df(50),
            lambda: pd.DataFrame(),
            "empty_test_only",
        ),
        (
            lambda: _nan_amount_df(50),
            lambda: _make_df(20),
            "nan_amount_in_train",
        ),
        (
            lambda: _string_amount_df(50),
            lambda: _make_df(20),
            "string_amount_in_train",
        ),
    ],
    ids=lambda x: x if isinstance(x, str) else "",
)
def test_preprocess_raises_on_invalid_input(get_train, get_test, description):
    with pytest.raises((ValueError, TypeError)):
        preprocess(get_train(), get_test())


class TestMainFunction:
    def test_main_creates_output_files(self, tmp_path):
        train_df = _make_df(100)
        test_df = _make_df(20)
        train_df.to_csv(tmp_path / "train.csv", index=False)
        test_df.to_csv(tmp_path / "test.csv", index=False)

        features_main(input_path=tmp_path, output_path=tmp_path, model_dir=tmp_path)

        assert (tmp_path / "train_featured.csv").exists()
        assert (tmp_path / "test_featured.csv").exists()
        assert (tmp_path / "scaler.pkl").exists()

    def test_main_file_not_found_raises(self, tmp_path):
        with pytest.raises(typer.Exit):
            features_main(
                input_path=tmp_path / "nonexistent",
                output_path=tmp_path,
                model_dir=tmp_path,
            )

    def test_main_output_is_scaled(self, tmp_path):
        train_df = _make_df(100)
        test_df = _make_df(20)
        train_df.to_csv(tmp_path / "train.csv", index=False)
        test_df.to_csv(tmp_path / "test.csv", index=False)

        features_main(input_path=tmp_path, output_path=tmp_path, model_dir=tmp_path)

        train_out = pd.read_csv(tmp_path / "train_featured.csv")
        for col in COLS_TO_SCALE:
            assert abs(train_out[col].mean()) < 0.1
