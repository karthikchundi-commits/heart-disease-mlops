import numpy as np
import pandas as pd

from heart import config
from heart.features import build_preprocessor


def test_preprocessor_output_has_no_nans(clean_df):
    X = build_preprocessor().fit_transform(clean_df[config.FEATURES])
    assert not np.isnan(X).any()
    assert X.shape[0] == len(clean_df)


def test_numeric_features_are_standardised(clean_df):
    prep = build_preprocessor().fit(clean_df[config.FEATURES])
    X = prep.transform(clean_df[config.FEATURES])
    names = list(prep.get_feature_names_out())
    age = X[:, names.index("num__age")]
    assert abs(age.mean()) < 1e-6 and abs(age.std() - 1) < 1e-6


def test_missing_indicator_added_for_ca(clean_df):
    names = build_preprocessor().fit(clean_df[config.FEATURES]).get_feature_names_out()
    assert "num__missingindicator_ca" in names


def test_fixed_one_hot_width_regardless_of_data(clean_df):
    prep = build_preprocessor().fit(clean_df[config.FEATURES])
    expected_cat = sum(len(v) + 1 for v in config.CATEGORY_LEVELS.values())
    n_cat = sum(n.startswith("cat__") for n in prep.get_feature_names_out())
    assert n_cat == expected_cat


def test_all_missing_optional_row_transforms(clean_df):
    prep = build_preprocessor().fit(clean_df[config.FEATURES])
    row = pd.DataFrame([{f: np.nan for f in config.FEATURES} | {"age": 50, "sex": 1, "cp": 4}])
    out = prep.transform(row[config.FEATURES])
    assert out.shape[0] == 1 and not np.isnan(out).any()
