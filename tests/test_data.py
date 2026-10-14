import pandas as pd

from src.data import MODEL_TO_BRAND, make_features


def test_every_model_has_brand():
    raw = pd.read_csv("data/raw/sneakerpulse_model_monthly_prices.csv")
    assert set(raw["model"]) <= set(MODEL_TO_BRAND)


def test_make_features_lag_and_target():
    df = pd.DataFrame({
        "model": ["A"] * 3,
        "brand": ["X"] * 3,
        "month": ["2024-01", "2024-02", "2024-03"],
        "median_price_usd": [100, 110, 120],
        "sales_recorded": [10, 20, 30],
        "brand_mean_price_usd": [105, 115, 125],
    })
    out = make_features(df)
    assert len(out) == 1                      # only the middle month remains
    assert out.iloc[0]["lag_price"] == 100    # previous month
    assert out.iloc[0]["target"] == 120       # next month