import pytest
from fastapi.testclient import TestClient

from heart import serving

PATIENT = {"age": 58, "sex": 1, "cp": 4, "trestbps": 140, "chol": 289, "fbs": 0, "restecg": 2,
           "thalach": 135, "exang": 1, "oldpeak": 1.8, "slope": 2, "ca": 1, "thal": 7}


@pytest.fixture
def client(fitted_pipeline, monkeypatch):
    monkeypatch.setattr(serving, "load_model", lambda: None)
    with TestClient(serving.app) as c:
        serving.STATE.update(model=fitted_pipeline, meta={"model_name": "test", "model_version": "0"})
        yield c
    serving.STATE.update(model=None, meta={})


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_ready(client):
    assert client.get("/ready").status_code == 200


def test_predict_returns_prediction_and_confidence(client):
    r = client.post("/predict", json=PATIENT)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert 0.5 <= body["confidence"] <= 1
    assert body["risk_band"] in ("low", "moderate", "high")
    assert "x-request-id" in r.headers


def test_predict_accepts_missing_optional_fields(client):
    assert client.post("/predict", json={"age": 50, "sex": 0, "cp": 3}).status_code == 200


@pytest.mark.parametrize("bad", [{"age": 58}, {**PATIENT, "cp": 9}, {**PATIENT, "age": -3},
                                 {**PATIENT, "thal": 5}])
def test_predict_rejects_invalid_input(client, bad):
    assert client.post("/predict", json=bad).status_code == 422


def test_batch(client):
    r = client.post("/predict/batch", json={"patients": [PATIENT, {**PATIENT, "cp": 2}]})
    assert r.status_code == 200 and len(r.json()["predictions"]) == 2


def test_metrics_exposed(client):
    client.post("/predict", json=PATIENT)
    text = client.get("/metrics").text
    assert "heart_api_requests_total" in text
    assert "heart_api_predictions_total" in text


def test_503_when_model_missing(client):
    serving.STATE["model"] = None
    assert client.post("/predict", json=PATIENT).status_code == 503
    assert client.get("/ready").status_code == 503
