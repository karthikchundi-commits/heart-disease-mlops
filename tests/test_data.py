import numpy as np
import pandas as pd

from heart import config
from heart.data import clean, load_raw


def test_target_is_binary(clean_df):
    assert set(clean_df[config.TARGET].unique()) <= {0, 1}
    assert "num" not in clean_df.columns


def test_target_binarisation(raw_df):
    out = clean(raw_df)
    expected = (raw_df.drop_duplicates()["num"] > 0).astype(int).to_numpy()
    assert np.array_equal(out[config.TARGET].to_numpy(), expected)


def test_zero_cholesterol_becomes_missing(raw_df):
    out = clean(raw_df)
    assert (out["chol"] == 0).sum() == 0
    assert out["chol"].isna().sum() >= 5


def test_invalid_category_codes_become_missing(raw_df):
    out = clean(raw_df)
    for col, levels in config.CATEGORY_LEVELS.items():
        assert out[col].dropna().isin(levels).all(), col


def test_output_schema(clean_df):
    assert list(clean_df.columns) == config.FEATURES + ["site", config.TARGET]


def test_duplicates_removed(raw_df):
    doubled = pd.concat([raw_df, raw_df.iloc[:5]], ignore_index=True)
    assert len(clean(doubled)) == len(clean(raw_df))


def test_clean_does_not_mutate_input(raw_df):
    before = raw_df.copy()
    clean(raw_df)
    assert raw_df.equals(before)


def test_load_raw_parses_question_marks(tmp_path):
    row = "63.0,1.0,1.0,145.0,233.0,1.0,2.0,150.0,0.0,2.3,3.0,?,6.0,0\n"
    for fname in config.SITES.values():
        (tmp_path / fname).write_text(row)
    df = load_raw(tmp_path)
    assert len(df) == len(config.SITES)
    assert df["ca"].isna().all()
    assert set(df["site"]) == set(config.SITES)
