import json

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Sneaker Resale Price API")
model = joblib.load("models/model.pkl")
with open("models/lookup.json") as f:
    lookup = json.load(f)

FEATURES = [
    "median_price_usd", "lag_price", "sales_recorded",
    "brand_mean_price_usd", "month_num",
    "price_change", "price_change_pct",
]

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


@app.post("/predict")
def predict(data: PriceInput):
    if data.model not in lookup:
        raise HTTPException(404, f"Unknown model: {data.model}")
    if data.lag_price <= 0:
        raise HTTPException(422, "lag_price must be greater than 0")
    
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
    return {"model": data.model,
            "predicted_next_month_price": round(price, 2),
            "predicted_change": round(delta, 2)}

