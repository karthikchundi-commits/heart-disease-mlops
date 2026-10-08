"""Integration tests: full training run (with MLflow) and EDA on synthetic data in a sandbox."""
import json

import pytest
from conftest import make_raw

from heart import config, eda, train
from heart.data import clean


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Point every output path at tmp_path so tests never touch real artefacts."""
    csv = tmp_path / "clean.csv"
    clean(make_raw(n=240, seed=1)).to_csv(csv, index=False)
    monkeypatch.setattr(config, "ROOT", tmp_path)
    monkeypatch.setattr(config, "CLEAN_CSV", csv)
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(config, "MODEL_PATH", tmp_path / "models" / "m.joblib")
    monkeypatch.setattr(config, "MODEL_META", tmp_path / "models" / "meta.json")
    monkeypatch.setattr(config, "FIGURES_DIR", tmp_path / "reports" / "figures")
    monkeypatch.setattr(eda, "OUT", tmp_path / "reports" / "figures" / "eda")
    (tmp_path / "reports").mkdir()
    # tiny grids keep the test fast
    small = {name: (est, {k: v[:1] for k, v in grid.items()})
             for name, (est, grid) in train.MODEL_SPECS.items()}
    monkeypatch.setattr(train, "MODEL_SPECS", small)
    monkeypatch.setattr(train, "N_JOBS", 1)
    return tmp_path


def test_train_end_to_end(sandbox):
    uri = f"sqlite:///{(sandbox / 'mlflow.db').as_posix()}"
    meta = train.main(["--tracking-uri", uri, "--experiment", "test-exp",
                       "--models", "logistic_regression", "random_forest"])
    assert (sandbox / "models" / "m.joblib").exists()
    saved = json.loads((sandbox / "models" / "meta.json").read_text())
    assert saved["model_name"] in ("logistic_regression", "random_forest")
    assert meta["metrics"]["cv_roc_auc"] > 0.5
    assert (sandbox / "reports" / "model_comparison.csv").exists()

    import mlflow
    mlflow.set_tracking_uri(uri)
    runs = mlflow.search_runs(experiment_names=["test-exp"])
    # 2 parent runs + 1 child each (1-candidate grids) + champion run
    assert len(runs) == 5
    assert {"metrics.cv_roc_auc", "metrics.test_recall"} <= set(runs.columns)
    version = mlflow.MlflowClient().get_model_version_by_alias("heart-disease-classifier", "champion")
    assert str(version.version) == "1"


def test_eda_writes_all_figures(sandbox):
    paths = eda.run_all()
    assert len(paths) == 6
    assert all((sandbox / "reports" / "figures" / "eda" / p.split("\\")[-1].split("/")[-1]).exists()
               for p in paths)
    assert (sandbox / "reports" / "eda_summary.csv").exists()
