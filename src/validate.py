"""Validate the model-ready table (data/processed/clean.csv) before training.

Run on its own:  uv run python -m src.validate
Training also calls validate() first and stops if any check fails.
"""
import sys

import pandas as pd

from src.data import CLEAN_PATH, MODEL_TO_BRAND

REQUIRED_COLUMNS = [
    "model", "brand", "month", "median_price_usd", "sales_recorded",
    "brand_mean_price_usd", "lag_price", "target", "month_num",

]

class DataValidationError(ValueError):

    """Raised when the data fails one or more checks."""
    
def check(df):

    """Return a list of problems found. An empty list means the data is fine."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return [f"missing columns: {missing}"]

    problems = []

    for col in ("median_price_usd", "lag_price", "target"):
        bad = int((df[col] <= 0).sum())
        if bad:
            problems.append(f"{col}: {bad} rows are not > 0")

    nulls = df[REQUIRED_COLUMNS].isna().sum()
    for col, n in nulls[nulls > 0].items():
        problems.append(f"{col}: {int(n)} null values")

    bad_months = ~df["month"].astype(str).str.fullmatch(r"\d{4}-(0[1-9]|1[0-2])")
    if bad_months.any():
        problems.append(f"month: {int(bad_months.sum())} values are not YYYY-MM")

    unknown = sorted(set(df["model"].dropna()) - set(MODEL_TO_BRAND))
    if unknown:
        problems.append(f"models not in MODEL_TO_BRAND: {unknown}")

    wrong_brand = df[df["model"].map(MODEL_TO_BRAND) != df["brand"]]
    if len(wrong_brand):
        problems.append(f"brand does not match MODEL_TO_BRAND in {len(wrong_brand)} rows")

    dupes = int(df.duplicated(["model", "month"]).sum())
    if dupes:
        problems.append(f"{dupes} duplicate (model, month) rows")

    return problems

def validate(df=None, path=CLEAN_PATH):
    """Load clean.csv (or use the given frame) and raise if any check fails."""
    if df is None:
        df = pd.read_csv(path)
    problems = check(df)
    if problems:
        raise DataValidationError(
            "Data validation failed:\n  - " + "\n  - ".join(problems))
    return df

if __name__ == "__main__":
    try:
        validate()
    except DataValidationError as e:
        print(e)
        sys.exit(1)
    print("Data validation passed")