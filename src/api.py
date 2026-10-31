import json
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import time
from fastapi import Request, Response
from prometheus_client import (Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST,)


app = FastAPI(title="Sneaker Resale Price API")
model = joblib.load("models/model.pkl")
with open("models/lookup.json") as f:
    lookup = json.load(f)

# Training-data stats written by src/train.py (used for drift monitoring)
try:
    with open("models/baseline_stats.json") as f:
        baseline = json.load(f)
except FileNotFoundError:

    baseline = None

FEATURES = [
    "median_price_usd", "lag_price", "sales_recorded",
    "brand_mean_price_usd", "month_num",
    "price_change", "price_change_pct",
]

REQUESTS = Counter("api_requests_total", "Total requests", ["path", "status"])
LATENCY = Histogram("api_request_latency_seconds", "Request latency", ["path"])
PREDICTED_PRICE = Histogram(
    "predicted_price_usd", "Predicted next-month price",
    buckets=[50, 100, 150, 200, 300, 500, 1000],
)

ERRORS = Counter("api_errors_total", "Responses with status 4xx/5xx", ["path", "status"])

# Drift: compare live inputs with the training data.
# Graph rate(input_median_price_usd_sum[5m]) / rate(input_median_price_usd_count[5m])
# next to training_median_price_mean in Grafana.
INPUT_PRICE = Histogram(
    "input_median_price_usd", "median_price_usd sent to /predict",
    buckets=[50, 100, 150, 200, 300, 500, 1000],
)
TRAIN_PRICE_MEAN = Gauge("training_median_price_mean", "Mean median_price_usd in training data")
TRAIN_PRICE_STD = Gauge("training_median_price_std", "Std of median_price_usd in training data")
if baseline:
    TRAIN_PRICE_MEAN.set(baseline["median_price_usd"]["mean"])
    TRAIN_PRICE_STD.set(baseline["median_price_usd"]["std"])

@app.middleware("http")
async def track_metrics(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    path = request.url.path
    if path != "/metrics":
        LATENCY.labels(path).observe(time.time() - start)
        REQUESTS.labels(path, str(response.status_code)).inc()

    if response.status_code >= 400:
        ERRORS.labels(path, str(response.status_code)).inc()
    return response

class PriceInput(BaseModel):
    model: str                   # e.g. "Samba"
    median_price_usd: float      # current month price
    lag_price: float             # previous month price
    sales_recorded: int          # current month sales
    brand_mean_price_usd: float  # brand average price
    month_num: int               # 1-12

@app.get("/")
def home():
    return {"message": "API is running", "models": sorted(lookup)}

@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": len(lookup)}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict")
def predict(data: PriceInput):
    if data.model not in lookup:
        raise HTTPException(404, f"Unknown model: {data.model}")
    if data.lag_price <= 0:
        raise HTTPException(422, "lag_price must be greater than 0")

    INPUT_PRICE.observe(data.median_price_usd)
    price_change = data.median_price_usd - data.lag_price
    row = pd.DataFrame([{
        "median_price_usd": data.median_price_usd,
        "lag_price": data.lag_price,
        "sales_recorded": data.sales_recorded,
        "brand_mean_price_usd": data.brand_mean_price_usd,
        "month_num": data.month_num,
        "price_change": price_change,
        "price_change_pct": price_change / data.lag_price,
    }])[FEATURES]

    delta = float(model.predict(row)[0])
    price = data.median_price_usd + delta
    PREDICTED_PRICE.observe(price)
    return {"model": data.model,
            "predicted_next_month_price": round(price, 2),
            "predicted_change": round(delta, 2)}

