"""Train, tune and compare classifiers with full MLflow tracking.

Run:  python -m heart.train
Each model family gets a parent MLflow run (best params, CV + hold-out
metrics, plots, logged model) and one nested child run per grid candidate.
The family with the best mean cross-validated ROC-AUC is promoted: it is
registered in the MLflow model registry and exported to models/ for serving.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import datetime, timezone

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split  # noqa: E402

from heart import __version__, config  # noqa: E402
from heart.data import load_clean  # noqa: E402
from heart.features import build_pipeline  # noqa: E402

log = logging.getLogger(__name__)
N_JOBS = int(os.getenv("HEART_N_JOBS", "-1"))  # parallel CV workers

SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}

MODEL_SPECS = {
    "logistic_regression": (
        LogisticRegression(solver="liblinear", max_iter=2000, random_state=config.RANDOM_STATE),
        {
            "model__C": [0.01, 0.1, 0.5, 1.0, 5.0],
            "model__penalty": ["l1", "l2"],
            "model__class_weight": [None, "balanced"],
        },
    ),
    "random_forest": (
        RandomForestClassifier(random_state=config.RANDOM_STATE),
        {
            "model__n_estimators": [200, 400],
            "model__max_depth": [None, 5, 8],
            "model__min_samples_leaf": [1, 4],
            "model__max_features": ["sqrt", 0.5],
        },
    ),
    "hist_gradient_boosting": (
        HistGradientBoostingClassifier(random_state=config.RANDOM_STATE),
        {
            "model__learning_rate": [0.03, 0.1],
            "model__max_depth": [3, None],
            "model__max_iter": [100, 250],
            "model__l2_regularization": [0.0, 1.0],
        },
    ),
}


def holdout_metrics(y_true, proba, threshold: float = 0.5) -> dict[str, float]:
    pred = (proba >= threshold).astype(int)
    return {
        "test_accuracy": accuracy_score(y_true, pred),
        "test_precision": precision_score(y_true, pred, zero_division=0),
        "test_recall": recall_score(y_true, pred, zero_division=0),
        "test_f1": f1_score(y_true, pred, zero_division=0),
        "test_roc_auc": roc_auc_score(y_true, proba),
    }


def _feature_names(pipeline) -> list[str]:
    return list(pipeline.named_steps["prep"].get_feature_names_out())


def plot_diagnostics(name, pipeline, X_test, y_test, out_dir) -> list[str]:
    """Confusion matrix, ROC, PR and feature-importance plots for one model."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    ConfusionMatrixDisplay.from_estimator(pipeline, X_test, y_test, ax=axes[0], cmap="Blues",
                                          display_labels=["No disease", "Disease"], colorbar=False)
    axes[0].set_title("Confusion matrix (hold-out)")
    RocCurveDisplay.from_estimator(pipeline, X_test, y_test, ax=axes[1], name=name)
    axes[1].plot([0, 1], [0, 1], ls="--", c="grey", lw=1)
    axes[1].set_title("ROC curve (hold-out)")
    PrecisionRecallDisplay.from_estimator(pipeline, X_test, y_test, ax=axes[2], name=name)
    axes[2].set_title("Precision-recall (hold-out)")
    fig.suptitle(name.replace("_", " ").title(), fontweight="bold")
    fig.tight_layout()
    p = out_dir / f"{name}_diagnostics.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths.append(str(p))

    model = pipeline.named_steps["model"]
    names = _feature_names(pipeline)
    if hasattr(model, "coef_"):
        values, label = model.coef_.ravel(), "coefficient"
    elif hasattr(model, "feature_importances_"):
        values, label = model.feature_importances_, "impurity importance"
    else:
        values, label = None, None
    if values is not None:
        s = pd.Series(values, index=names).sort_values(key=np.abs).tail(15)
        fig, ax = plt.subplots(figsize=(7, 5.5))
        s.plot.barh(ax=ax, color=np.where(s >= 0, "#c0392b", "#2471a3"))
        ax.set_xlabel(label)
        ax.set_title(f"Top features - {name}")
        fig.tight_layout()
        p = out_dir / f"{name}_features.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        paths.append(str(p))
    return paths


def train_family(name, estimator, grid, X_train, y_train, X_test, y_test, cv) -> dict:
    with mlflow.start_run(run_name=name) as parent:
        mlflow.set_tags({"model_family": name, "stage": "tuning"})
        search = GridSearchCV(build_pipeline(estimator), grid, scoring=SCORING, refit="roc_auc",
                              cv=cv, n_jobs=N_JOBS, return_train_score=False)
        t0 = time.perf_counter()
        search.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - t0

        res = pd.DataFrame(search.cv_results_)
        for i, row in res.iterrows():
            with mlflow.start_run(run_name=f"{name}-cand{i:02d}", nested=True):
                mlflow.set_tag("model_family", name)
                mlflow.log_params({k.replace("model__", ""): v for k, v in row["params"].items()})
                mlflow.log_metrics({f"cv_{m}": row[f"mean_test_{m}"] for m in SCORING})
                mlflow.log_metric("cv_roc_auc_std", row["std_test_roc_auc"])
                mlflow.log_metric("rank_roc_auc", int(row["rank_test_roc_auc"]))

        best = search.best_estimator_
        b = search.best_index_
        cv_metrics = {f"cv_{m}": float(res.loc[b, f"mean_test_{m}"]) for m in SCORING}
        cv_metrics["cv_roc_auc_std"] = float(res.loc[b, "std_test_roc_auc"])
        proba = best.predict_proba(X_test)[:, 1]
        test_metrics = holdout_metrics(y_test, proba)

        mlflow.log_params({k.replace("model__", "best_"): v for k, v in search.best_params_.items()})
        mlflow.log_params({"n_candidates": len(res), "cv_folds": cv.get_n_splits(),
                           "n_train": len(X_train), "n_test": len(X_test)})
        mlflow.log_metrics({**cv_metrics, **test_metrics, "fit_seconds": fit_seconds})

        out_dir = config.FIGURES_DIR / "models"
        for p in plot_diagnostics(name, best, X_test, y_test, out_dir):
            mlflow.log_artifact(p, artifact_path="plots")
        cv_csv = out_dir / f"{name}_cv_results.csv"
        res.drop(columns="params").to_csv(cv_csv, index=False)
        mlflow.log_artifact(str(cv_csv), artifact_path="cv")

        signature = infer_signature(X_test, best.predict_proba(X_test))
        mlflow.sklearn.log_model(best, artifact_path="model", signature=signature,
                                 input_example=X_test.head(3))
        log.info("%-24s cv_roc_auc=%.4f (+/-%.4f)  test_roc_auc=%.4f  candidates=%d  %.1fs",
                 name, cv_metrics["cv_roc_auc"], cv_metrics["cv_roc_auc_std"],
                 test_metrics["test_roc_auc"], len(res), fit_seconds)
        return {"name": name, "run_id": parent.info.run_id, "pipeline": best,
                "best_params": search.best_params_, **cv_metrics, **test_metrics}


def compare_plot(results, X_test, y_test) -> str:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for r in results:
        RocCurveDisplay.from_estimator(r["pipeline"], X_test, y_test, ax=ax, name=r["name"])
    ax.plot([0, 1], [0, 1], ls="--", c="grey", lw=1)
    ax.set_title("Hold-out ROC - model comparison")
    fig.tight_layout()
    p = config.FIGURES_DIR / "model_comparison_roc.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return str(p)


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description="Train and track heart disease classifiers")
    default_uri = f"sqlite:///{(config.ROOT / 'mlflow.db').as_posix()}"
    parser.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI", default_uri))
    parser.add_argument("--experiment", default="heart-disease-classification")
    parser.add_argument("--models", nargs="+", default=list(MODEL_SPECS), choices=list(MODEL_SPECS))
    parser.add_argument("--no-register", action="store_true", help="skip the MLflow model registry")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    mlflow.set_tracking_uri(args.tracking_uri)
    if mlflow.get_experiment_by_name(args.experiment) is None:
        mlflow.create_experiment(args.experiment, artifact_location=(config.ROOT / "mlruns").as_uri())
    mlflow.set_experiment(args.experiment)

    df = load_clean()
    X, y = df[config.FEATURES], df[config.TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=config.RANDOM_STATE)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)
    log.info("train=%d test=%d features=%d", len(X_train), len(X_test), X.shape[1])

    results = [train_family(n, *MODEL_SPECS[n], X_train, y_train, X_test, y_test, cv)
               for n in args.models]

    # Model selection uses CV score only - the hold-out set is never used to choose.
    champion = max(results, key=lambda r: r["cv_roc_auc"])
    cols = ["name", "cv_roc_auc", "cv_roc_auc_std", "cv_accuracy", "cv_precision", "cv_recall", "cv_f1",
            "test_roc_auc", "test_accuracy", "test_precision", "test_recall", "test_f1"]
    table = pd.DataFrame(results)[cols].sort_values("cv_roc_auc", ascending=False)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(config.ROOT / "reports" / "model_comparison.csv", index=False)
    roc_png = compare_plot(results, X_test, y_test)
    log.info("model comparison:\n%s", table.round(4).to_string(index=False))

    # Refit the champion on all labelled data for the deployable artefact.
    final = champion["pipeline"]
    final.fit(X, y)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, config.MODEL_PATH)
    meta = {
        "model_name": champion["name"],
        "model_version": __version__,
        "mlflow_run_id": champion["run_id"],
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "features": config.FEATURES,
        "best_params": {k.replace("model__", ""): v for k, v in champion["best_params"].items()},
        "metrics": {k: round(float(champion[k]), 4) for k in cols[1:]},
        "n_training_rows": len(X),
        "threshold": 0.5,
    }
    config.MODEL_META.write_text(json.dumps(meta, indent=2, default=str))

    with mlflow.start_run(run_name="champion-final"):
        mlflow.set_tags({"model_family": champion["name"], "stage": "production-candidate"})
        mlflow.log_params({"champion": champion["name"], "selected_by": "cv_roc_auc",
                           "n_training_rows": len(X)})
        mlflow.log_metrics({k: float(champion[k]) for k in cols[1:]})
        mlflow.log_artifact(roc_png, artifact_path="plots")
        mlflow.log_artifact(str(config.ROOT / "reports" / "model_comparison.csv"))
        mlflow.log_artifact(str(config.MODEL_META))
        info = mlflow.sklearn.log_model(
            final, artifact_path="model", input_example=X_test.head(3),
            signature=infer_signature(X_test, final.predict_proba(X_test)),
            registered_model_name=None if args.no_register else "heart-disease-classifier")
    if not args.no_register:
        version = info.registered_model_version
        mlflow.MlflowClient().set_registered_model_alias("heart-disease-classifier", "champion", version)
        log.info("registered heart-disease-classifier v%s as @champion", version)

    log.info("champion=%s  saved %s", champion["name"], config.MODEL_PATH)
    return meta


if __name__ == "__main__":
    main()
