"""Replay realistic requests against the API so Prometheus/Grafana have data.

Samples real patients from the cleaned dataset, and mixes in a small share of
invalid payloads so 4xx panels are populated too.

    python scripts/generate_traffic.py --base-url http://localhost:8080 --requests 300
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

import pandas as pd

FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca",
            "sex", "fbs", "exang", "cp", "restecg", "slope", "thal"]
INT_FIELDS = {"sex", "fbs", "exang", "cp", "restecg", "slope", "thal", "ca"}
DATA = Path(__file__).resolve().parents[1] / "data" / "processed" / "heart_clean.csv"


def to_payload(row: pd.Series) -> dict:
    out = {}
    for f in FEATURES:
        v = row[f]
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        out[f] = int(v) if f in INT_FIELDS else float(v)
    return out


def post(url: str, payload: dict) -> int:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--requests", type=int, default=300)
    ap.add_argument("--invalid-share", type=float, default=0.05)
    ap.add_argument("--sleep", type=float, default=0.05)
    args = ap.parse_args()

    df = pd.read_csv(DATA)
    rng = random.Random(7)
    statuses: Counter = Counter()
    t0 = time.time()
    for _ in range(args.requests):
        if rng.random() < args.invalid_share:
            statuses[post(f"{args.base_url}/predict", {"age": 300, "sex": 1, "cp": 9})] += 1
        else:
            row = df.iloc[rng.randrange(len(df))]
            statuses[post(f"{args.base_url}/predict", to_payload(row))] += 1
        time.sleep(args.sleep)
    print(f"sent {args.requests} requests in {time.time() - t0:.1f}s  status counts: {dict(statuses)}")


if __name__ == "__main__":
    main()
