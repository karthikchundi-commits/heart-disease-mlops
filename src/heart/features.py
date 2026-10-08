"""Feature engineering: one sklearn ColumnTransformer shared by training and serving."""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from heart import config


def build_preprocessor() -> ColumnTransformer:
    """Imputation + scaling + encoding.

    numeric     : median impute, add a was-missing flag (missingness itself is
                  informative here - e.g. 'ca' is only recorded at Cleveland),
                  then standardise.
    binary      : most-frequent impute, passed through as 0/1.
    categorical : missing becomes its own level, then one-hot encode with
                  a fixed category list so unseen codes at serving time map
                  to all-zeros instead of raising.
    """
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    binary = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value=-1)),
        ("onehot", OneHotEncoder(
            categories=[[-1] + config.CATEGORY_LEVELS[c] for c in config.CATEGORICAL_FEATURES],
            handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric, config.NUMERIC_FEATURES),
        ("bin", binary, config.BINARY_FEATURES),
        ("cat", categorical, config.CATEGORICAL_FEATURES),
    ])


def build_pipeline(estimator) -> Pipeline:
    return Pipeline([("prep", build_preprocessor()), ("model", estimator)])
