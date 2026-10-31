import pandas as pd
import pytest

from src.validate import DataValidationError, check, validate

def good_frame():
    return pd.DataFrame({
        "model": ["Dunk", "Dunk"],
        "brand": ["Nike", "Nike"],
        "month": ["2024-01", "2024-02"],
        "median_price_usd": [100.0, 110.0],
        "sales_recorded": [10, 20],
        "brand_mean_price_usd": [105.0, 115.0],
        "lag_price": [95.0, 100.0],
        "target": [110.0, 120.0],
        "month_num": [1, 2],
    })

def test_good_data_passes():
    assert check(good_frame()) == []

def test_non_positive_price_fails():
    df = good_frame()
    df.loc[0, "median_price_usd"] = 0
    assert any("median_price_usd" in p for p in check(df))

def test_null_month_fails():
    df = good_frame()
    df.loc[0, "month"] = None
    assert any("month" in p for p in check(df))

def test_unknown_model_fails():
    df = good_frame()
    df.loc[0, "model"] = "Not A Shoe"
    assert any("MODEL_TO_BRAND" in p for p in check(df))

def test_validate_raises_on_bad_data():
    df = good_frame()
    df.loc[0, "target"] = -5
    with pytest.raises(DataValidationError):
        validate(df)

def test_real_clean_csv_passes():
    validate() # data/processed/clean.csv is tracked in Git
    