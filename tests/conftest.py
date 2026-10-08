"""Shared fixtures. Tests use a synthetic UCI-shaped dataset so they run offline and fast."""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from heart import config
from heart.data import clean
from heart.features import build_pipeline


def make_raw(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "age": rng.integers(29, 78, n).astype(float),
        "sex": rng.integers(0, 2, n).astype(float),
        "cp": rng.integers(1, 5, n).astype(float),
        "trestbps": rng.normal(132, 18, n).round(),
        "chol": rng.normal(240, 50, n).round(),
        "fbs": rng.integers(0, 2, n).astype(float),
        "restecg": rng.integers(0, 3, n).astype(float),
        "thalach": rng.normal(140, 25, n).round(),
        "exang": rng.integers(0, 2, n).astype(float),
        "oldpeak": rng.gamma(1.2, 0.9, n).round(1),
        "slope": rng.integers(1, 4, n).astype(float),
        "ca": rng.integers(0, 4, n).astype(float),
        "thal": rng.choice([3.0, 6.0, 7.0], n),
    })
    # outcome loosely driven by known risk factors so models have signal
    logit = 0.9 * (df["cp"] == 4) + 0.8 * df["exang"] + 0.6 * df["oldpeak"] - 0.02 * (df["thalach"] - 140)
    df["num"] = np.where(rng.random(n) < 1 / (1 + np.exp(-(logit - 1))), rng.integers(1, 5, n), 0)
    # inject the kinds of dirt seen in the real files
    df.loc[0:9, "ca"] = np.nan
    df.loc[10:14, "chol"] = 0
    df.loc[15:17, "thal"] = 5  # invalid code
    df["site"] = "synthetic"
    return df


@pytest.fixture
def raw_df() -> pd.DataFrame:
    return make_raw()


@pytest.fixture
def clean_df(raw_df) -> pd.DataFrame:
    return clean(raw_df)


@pytest.fixture
def fitted_pipeline(clean_df):
    pipe = build_pipeline(LogisticRegression(max_iter=1000))
    pipe.fit(clean_df[config.FEATURES], clean_df[config.TARGET])
    return pipe
