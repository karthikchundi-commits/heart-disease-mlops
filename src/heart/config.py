"""Central configuration: paths, column schema and feature groups."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
CLEAN_CSV = DATA_PROCESSED / "heart_clean.csv"
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "heart_model.joblib"
MODEL_META = MODELS_DIR / "model_meta.json"
FIGURES_DIR = ROOT / "reports" / "figures"

UCI_BASE = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/"

# The four collection sites of the UCI Heart Disease study. All four are used
# (920 patients) rather than Cleveland alone (303) - this gives more data but
# also real-world missingness that the pipeline has to handle.
SITES = {
    "cleveland": "processed.cleveland.data",
    "hungarian": "processed.hungarian.data",
    "switzerland": "processed.switzerland.data",
    "va": "processed.va.data",
}

RAW_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]

TARGET = "target"

NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
BINARY_FEATURES = ["sex", "fbs", "exang"]
CATEGORICAL_FEATURES = ["cp", "restecg", "slope", "thal"]
FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES

# Valid category codes (from the UCI data dictionary).
CATEGORY_LEVELS = {
    "cp": [1, 2, 3, 4],
    "restecg": [0, 1, 2],
    "slope": [1, 2, 3],
    "thal": [3, 6, 7],
}

RANDOM_STATE = 42
