"""Offline inference from the saved model artefact.

Run:  python -m heart.predict samples/patient_high_risk.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from heart import config


def predict_records(records: list[dict], model_path: Path = config.MODEL_PATH) -> list[dict]:
    model = joblib.load(model_path)
    frame = pd.DataFrame(records).reindex(columns=config.FEATURES).astype(float)
    proba = model.predict_proba(frame)[:, 1]
    return [{"prediction": int(p >= 0.5), "probability_disease": round(float(p), 4),
             "confidence": round(float(max(p, 1 - p)), 4)} for p in proba]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Predict heart disease for patients in a JSON file")
    parser.add_argument("json_file", type=Path, help="one patient object or a list of them")
    parser.add_argument("--model", type=Path, default=config.MODEL_PATH)
    args = parser.parse_args(argv)
    data = json.loads(args.json_file.read_text())
    records = data if isinstance(data, list) else [data]
    for rec, out in zip(records, predict_records(records, args.model), strict=True):
        print(json.dumps({"input": rec, **out}))


if __name__ == "__main__":
    main()
