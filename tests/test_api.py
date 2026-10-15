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