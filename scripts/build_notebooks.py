"""Generate the three project notebooks (run once; they are then executed by nbconvert)."""
from pathlib import Path

import nbformat as nbf

NB_DIR = Path(__file__).resolve().parents[1] / "notebooks"
SETUP = """import os, sys, warnings
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
from pathlib import Path
warnings.filterwarnings("ignore")
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))
import pandas as pd, numpy as np
import matplotlib.pyplot as plt, seaborn as sns
from IPython.display import Image, display
from heart import config
pd.set_option("display.width", 160)
sns.set_theme(style="whitegrid")"""


def md(s):
    return nbf.v4.new_markdown_cell(s.strip())


def code(s):
    return nbf.v4.new_code_cell(s.strip())


def write(name, cells):
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nbf.write(nb, NB_DIR / name)
    print("wrote", name)


eda = [
    md("""# 01 - Data acquisition & exploratory data analysis
**Dataset:** UCI Heart Disease, all four collection sites (Cleveland, Hungary, Switzerland,
VA Long Beach) = 920 patients, 13 clinical features, diagnosis `num` 0-4.

Design decision: most published solutions use the 303-row Cleveland file only. Using all four
sites triples the data and forces the pipeline to deal with the **real missingness** that a
production system would face (several sites never measured `ca`, `thal`, `slope`)."""),
    code(SETUP),
    md("## 1. Acquire\n`heart.data.download()` fetches the four `processed.*.data` files (cached in `data/raw/`)."),
    code("""from heart.data import download, load_raw, clean
paths = download()
raw = load_raw()
print(raw.shape)
raw.head()"""),
    md("## 2. Data quality audit of the raw files"),
    code("""audit = pd.DataFrame({
    "missing ('?')": raw.isna().sum(),
    "zeros": (raw == 0).sum(),
    "dtype": raw.dtypes.astype(str),
})
audit.loc[["chol", "trestbps", "ca", "thal", "slope"]]"""),
    code("""print("rows with cholesterol == 0 by site:")
print(raw.loc[raw.chol == 0, "site"].value_counts())
print("\\nduplicate rows:", raw.duplicated().sum())"""),
    md("""**Findings**
* `?` is the missing marker - parsed to NaN.
* `chol == 0` (172 rows: 123 Switzerland, 49 VA - Switzerland never recorded cholesterol) and `trestbps == 0` are physiologically impossible -> treated as missing.
* `num` is 0-4 severity; the task is binary, so target = `num > 0`.
* 2 exact duplicate rows are dropped.

Imputation is **not** done here - it lives inside the sklearn pipeline so the identical logic runs at serving time."""),
    md("## 3. Clean"),
    code("""df = clean(raw)
df.to_csv(config.CLEAN_CSV, index=False)
print(df.shape, "positive rate = %.1f%%" % (100 * df.target.mean()))
df.describe().T.round(2)"""),
    md("## 4. Visual EDA\nAll figures are produced by `heart.eda` (also run in CI) and saved to `reports/figures/eda/`."),
    code("""from heart import eda
figs = eda.run_all()
for f in figs:
    display(Image(f))"""),
    md("""## 5. Key observations
| Observation | Consequence for modelling |
|---|---|
| Classes are reasonably balanced overall (55 % / 45 %) but prevalence differs strongly by site (Switzerland ~93 % positive) | Stratified splits; `site` is **excluded** from features - it would leak collection-centre prevalence and is not available at inference time |
| `ca` 66 %, `thal` 53 %, `slope` 33 % missing, concentrated in non-Cleveland sites | Median / constant imputation **plus missing-indicator flags** - missingness itself carries signal |
| Asymptomatic chest pain (`cp = 4`), exercise angina, ST depression (`oldpeak`), flat/down ST slope are the strongest univariate risk factors | Kept as-is; categorical codes one-hot encoded |
| `thalach` (max heart rate) is negatively correlated with disease | Linear models should do well |
| Skewed `oldpeak`, outliers in `chol` | Standard scaling; tree models are robust to the rest |"""),
    code("""df.groupby("site")[["age", "chol", "thalach", "oldpeak", "target"]].mean().round(2)"""),
]

training = [
    md("""# 02 - Feature engineering, model selection & experiment tracking
The production training entry-point is `python -m heart.train` (also executed by the CI
pipeline). This notebook documents *why* the final configuration was chosen and summarises
the MLflow runs."""),
    code(SETUP),
    md("## 1. Feature pipeline\nOne `ColumnTransformer` (`heart.features.build_preprocessor`) is shared by training and serving."),
    code("""from heart.features import build_preprocessor
from heart.data import load_clean
df = load_clean()
X, y = df[config.FEATURES], df[config.TARGET]
prep = build_preprocessor().fit(X)
Xt = prep.transform(X)
print("input features :", X.shape[1])
print("model features :", Xt.shape[1])
pd.Series(prep.get_feature_names_out()).to_frame("engineered feature").T"""),
    md("""| group | columns | steps |
|---|---|---|
| numeric | age, trestbps, chol, thalach, oldpeak, ca | median impute + missing indicator -> StandardScaler |
| binary | sex, fbs, exang | most-frequent impute |
| categorical | cp, restecg, slope, thal | missing -> own level (-1) -> one-hot with fixed category list |"""),
    md("## 2. Baseline comparison (default hyper-parameters, 5-fold stratified CV)"),
    code("""from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from heart.features import build_pipeline

cv = StratifiedKFold(5, shuffle=True, random_state=config.RANDOM_STATE)
baselines = {
    "majority class": DummyClassifier(strategy="most_frequent"),
    "logistic regression": LogisticRegression(max_iter=2000),
    "random forest": RandomForestClassifier(random_state=42),
    "hist gradient boosting": HistGradientBoostingClassifier(random_state=42),
}
rows = []
for name, est in baselines.items():
    s = cross_validate(build_pipeline(est), X, y, cv=cv,
                       scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])
    rows.append({"model": name, **{k[5:]: v.mean() for k, v in s.items() if k.startswith("test_")}})
pd.DataFrame(rows).set_index("model").round(3)"""),
    md("## 3. Does the missing-indicator design help?"),
    code("""from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def prep_variant(indicator):
    num = Pipeline([("i", SimpleImputer(strategy="median", add_indicator=indicator)), ("s", StandardScaler())])
    cat = Pipeline([("i", SimpleImputer(strategy="most_frequent")), ("o", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("n", num, config.NUMERIC_FEATURES),
                              ("b", SimpleImputer(strategy="most_frequent"), config.BINARY_FEATURES),
                              ("c", cat, config.CATEGORICAL_FEATURES)])

for label, p in [("no indicators, mode-imputed categories", prep_variant(False)),
                 ("with indicators, mode-imputed categories", prep_variant(True)),
                 ("final: indicators + 'missing' category", build_preprocessor())]:
    auc = cross_validate(Pipeline([("p", p), ("m", LogisticRegression(max_iter=2000))]), X, y,
                         cv=cv, scoring="roc_auc")["test_score"]
    print(f"{label:45s} ROC-AUC {auc.mean():.4f} +/- {auc.std():.4f}")"""),
    md("""## 4. Hyper-parameter tuning (what `heart.train` does)
* 80/20 stratified hold-out split; the hold-out set is **never** used for model choice.
* `GridSearchCV`, 5-fold stratified CV, refit on **ROC-AUC** (threshold-independent, robust to the site-level prevalence shift); accuracy, precision, recall and F1 are tracked too.
* Grids: LR (C x penalty x class_weight = 20), RF (n_estimators x depth x min_samples_leaf x max_features = 24), HGB (learning_rate x depth x iterations x L2 = 16).
* Champion = best mean CV ROC-AUC -> refit on all 918 rows -> `models/heart_model.joblib` + MLflow registry alias `@champion`."""),
    md("## 5. Experiment tracking summary (MLflow)"),
    code("""import mlflow
mlflow.set_tracking_uri(f"sqlite:///{(config.ROOT / 'mlflow.db').as_posix()}")
runs = mlflow.search_runs(experiment_names=["heart-disease-classification"])
print("total runs logged:", len(runs))
parents = runs[runs["tags.mlflow.parentRunId"].isna() & runs["tags.stage"].eq("tuning")]
cols = ["tags.mlflow.runName", "metrics.cv_roc_auc", "metrics.cv_roc_auc_std", "metrics.cv_f1",
        "metrics.test_roc_auc", "metrics.test_accuracy", "metrics.test_recall", "params.n_candidates"]
parents[cols].sort_values("metrics.cv_roc_auc", ascending=False).round(4)"""),
    code("""children = runs[runs["tags.mlflow.parentRunId"].notna()]
fig, ax = plt.subplots(figsize=(8, 4))
sns.stripplot(data=children, x="metrics.cv_roc_auc", y="tags.model_family", hue="tags.model_family",
              size=7, ax=ax, legend=False)
ax.set_title("CV ROC-AUC of every grid candidate (one MLflow child run each)")
ax.set_xlabel("mean 5-fold ROC-AUC"); ax.set_ylabel("")
plt.show()"""),
    code("""lr = children[children["tags.model_family"] == "logistic_regression"]
pivot = lr.pivot_table(index="params.C", columns=["params.penalty", "params.class_weight"],
                       values="metrics.cv_roc_auc")
sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis")
plt.title("Logistic regression grid - CV ROC-AUC"); plt.show()"""),
    md("## 6. Hold-out diagnostics of the tuned models"),
    code("""display(Image(str(config.FIGURES_DIR / "model_comparison_roc.png")))
for name in ["logistic_regression", "random_forest", "hist_gradient_boosting"]:
    display(Image(str(config.FIGURES_DIR / "models" / f"{name}_diagnostics.png")))"""),
    code("""import json
meta = json.loads(config.MODEL_META.read_text())
print(json.dumps(meta, indent=2))"""),
    md("""## 7. Model selection rationale
* All three tuned families land within ~0.01 CV ROC-AUC of each other - the CV standard deviation (~0.05) is larger than the gap.
* **Logistic regression (L1, C = 0.5)** has the best mean CV ROC-AUC, is the simplest, calibrates well, trains in milliseconds and gives directly interpretable coefficients (important in a clinical setting), so it is the champion.
* The L1 penalty zeroes out 8 of the 31 engineered features (e.g. all `restecg` levels, `thal = 6`) - a built-in feature selection.
* Recall (~0.86-0.89) is prioritised over precision in the report because missing a diseased patient is costlier than a false alarm; the API exposes the raw probability and a risk band so the threshold can be tuned downstream."""),
    code("""display(Image(str(config.FIGURES_DIR / "models" / "logistic_regression_features.png")))"""),
]

inference = [
    md("""# 03 - Inference
Three ways to use the trained model:
1. load the joblib pipeline directly,
2. load the registered MLflow model (`models:/heart-disease-classifier@champion`),
3. call the REST API (`/predict`)."""),
    code(SETUP),
    code("""import json, joblib
model = joblib.load(config.MODEL_PATH)
model"""),
    code("""patients = [json.loads(Path(f"samples/{n}.json").read_text())
            for n in ["patient_high_risk", "patient_low_risk", "patient_partial"]]
frame = pd.DataFrame(patients).reindex(columns=config.FEATURES).astype(float)
frame.assign(p_disease=model.predict_proba(frame)[:, 1].round(4), prediction=model.predict(frame))"""),
    md("Note the third patient has no `chol`, `ca`, `thal`, `slope` - the pipeline imputes them exactly as in training."),
    md("## MLflow model registry"),
    code("""import mlflow
mlflow.set_tracking_uri(f"sqlite:///{(config.ROOT / 'mlflow.db').as_posix()}")
registered = mlflow.sklearn.load_model("models:/heart-disease-classifier@champion")
np.allclose(registered.predict_proba(frame), model.predict_proba(frame))"""),
    md("## REST API (in-process test client - identical code path to the container)"),
    code("""from fastapi.testclient import TestClient
from heart.serving import app
with TestClient(app) as client:
    for p in patients:
        r = client.post("/predict", json=p)
        print(r.status_code, r.json())
    bad = client.post("/predict", json={"age": 58, "cp": 9})
    print(bad.status_code, bad.json()["detail"][0]["msg"])"""),
    md("""Against the running container / Kubernetes service:
```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \\
     -d @samples/patient_high_risk.json
```"""),
]

if __name__ == "__main__":
    NB_DIR.mkdir(exist_ok=True)
    write("01_eda.ipynb", eda)
    write("02_model_training.ipynb", training)
    write("03_inference.ipynb", inference)
