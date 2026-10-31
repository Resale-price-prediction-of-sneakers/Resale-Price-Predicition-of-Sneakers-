from fastapi.testclient import TestClient
from src.api import app, lookup

client = TestClient(app)

def _valid_payload(model_name):
    return {
        "model": model_name,
        "median_price_usd": 110,
        "lag_price": 100,
        "sales_recorded": 20,
        "brand_mean_price_usd": 115,
        "month_num": 2,
    }

def test_home_lists_models():
    r = client.get("/")
    assert r.status_code == 200
    assert "models" in r.json()

def test_predict_known_model():
    model_name = sorted(lookup)[0]
    r = client.post("/predict", json=_valid_payload(model_name))
    assert r.status_code == 200
    body = r.json()
    assert body["model"] == model_name
    assert "predicted_next_month_price" in body

def test_predict_unknown_model_returns_404():
    r = client.post("/predict", json=_valid_payload("NOT_A_MODEL"))
    assert r.status_code == 404

def test_predict_missing_field_returns_422():
    r = client.post("/predict", json={"model": "x"})
    assert r.status_code == 422

def test_predict_invalid_type_returns_422():
    payload = _valid_payload(sorted(lookup)[0])
    payload["lag_price"] = "not a number"
    r = client.post("/predict", json=payload)
    assert r.status_code == 422

def test_predict_returns_positive_price():
    model_name = sorted(lookup)[0]
    r = client.post("/predict", json=_valid_payload(model_name))
    assert r.json()["predicted_next_month_price"] > 0

def test_metrics_include_monitoring_metrics():
    client.post("/predict", json=_valid_payload(sorted(lookup)[0]))
    text = client.get("/metrics").text
    assert "api_requests_total" in text
    assert "api_request_latency_seconds" in text
    assert "input_median_price_usd_count" in text
    assert "training_median_price_mean" in text

def test_errors_are_counted():
    client.post("/predict", json=_valid_payload("NOT_A_MODEL"))
    text = client.get("/metrics").text
    assert 'api_errors_total{path="/predict",status="404"}' in text