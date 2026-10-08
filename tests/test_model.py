import json

import joblib
import numpy as np
import pytest
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

from heart import config
from heart.features import build_pipeline
from heart.predict import predict_records
from heart.train import MODEL_SPECS, holdout_metrics


def test_pipeline_learns_signal(fitted_pipeline, clean_df):
    proba = fitted_pipeline.predict_proba(clean_df[config.FEATURES])[:, 1]
    assert roc_auc_score(clean_df[config.TARGET], proba) > 0.65


def test_probabilities_are_valid(fitted_pipeline, clean_df):
    proba = fitted_pipeline.predict_proba(clean_df[config.FEATURES])
    assert np.all((proba >= 0) & (proba <= 1))
    assert np.allclose(proba.sum(axis=1), 1)


def test_holdout_metrics_keys_and_ranges():
    y = np.array([0, 0, 1, 1])
    m = holdout_metrics(y, np.array([0.1, 0.6, 0.4, 0.9]))
    assert set(m) == {"test_accuracy", "test_precision", "test_recall", "test_f1", "test_roc_auc"}
    assert m["test_accuracy"] == 0.5 and m["test_roc_auc"] == 0.75


@pytest.mark.parametrize("name", list(MODEL_SPECS))
def test_every_model_family_fits(name, clean_df):
    estimator, grid = MODEL_SPECS[name]
    pipe = build_pipeline(clone(estimator))
    pipe.set_params(**{k: v[0] for k, v in grid.items()})
    pipe.fit(clean_df[config.FEATURES], clean_df[config.TARGET])
    assert pipe.predict(clean_df[config.FEATURES].head(5)).shape == (5,)


def test_model_roundtrip_and_predict_records(fitted_pipeline, tmp_path):
    path = tmp_path / "m.joblib"
    joblib.dump(fitted_pipeline, path)
    out = predict_records([{"age": 60, "sex": 1, "cp": 4}], path)
    assert out[0]["prediction"] in (0, 1)
    assert 0.5 <= out[0]["confidence"] <= 1


@pytest.mark.skipif(not config.MODEL_META.exists(), reason="trained artefact not present")
def test_saved_artifact_meets_quality_bar():
    meta = json.loads(config.MODEL_META.read_text())
    assert meta["metrics"]["cv_roc_auc"] >= 0.80
    assert meta["features"] == config.FEATURES
