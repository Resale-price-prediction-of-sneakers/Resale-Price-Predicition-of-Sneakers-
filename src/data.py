"""Clean and merge the 3 SneakerPulse CSV files.

Grain of the merged table: one row per (model, month).

Outputs
  data/processed/merged.csv  -> full cleaned + merged table (all months)
  data/processed/clean.csv   -> model-ready table (lag features + next-month target)
"""
import pandas as pd

RAW_DIR = "data/raw"
MERGED_PATH = "data/processed/merged.csv"
CLEAN_PATH = "data/processed/clean.csv"

# The files do not say which brand a model belongs to, so we map it by hand.
MODEL_TO_BRAND = {
    "2002R": "New Balance", "574": "New Balance", "990": "New Balance",
    "Air Force 1": "Nike", "Air Max 1": "Nike", "Air Max 270": "Nike",
    "Air Max 90": "Nike", "Air Max 95": "Nike", "Air Max 97": "Nike",
    "Air Max Plus": "Nike", "Air VaporMax": "Nike", "Blazer": "Nike",
    "Dunk": "Nike", "Dunk SB": "Nike",
    "Air Jordan 1": "Air Jordan", "Air Jordan 11": "Air Jordan",
    "Air Jordan 4": "Air Jordan",
    "Chuck 70": "Converse",
    "Forum": "adidas", "Gazelle": "adidas", "NMD Runner": "adidas",
    "Samba": "adidas", "Superstar": "adidas", "UltraBoost": "adidas",
    "Yeezy Boost 350": "Yeezy", "Yeezy Slide": "Yeezy",
}


def load_raw():
    brand = pd.read_csv(f"{RAW_DIR}/brand_monthly.csv")
    model = pd.read_csv(f"{RAW_DIR}/model_monthly_prices.csv")
    shoe = pd.read_csv(f"{RAW_DIR}/same_shoe_12m.csv")
    return brand, model, shoe


def clean(brand, model, shoe):
    """Strip text, parse dates, drop duplicates, check ranges."""
    for df, col in [(brand, "brand"), (model, "model"), (shoe, "name")]:
        df[col] = df[col].str.strip()
    brand["month"] = pd.to_datetime(brand["month"], format="%Y-%m")
    model["month"] = pd.to_datetime(model["month"], format="%Y-%m")
    brand = brand.drop_duplicates(["month", "brand"])
    model = model.drop_duplicates(["model", "month"])

    assert model["median_price_usd"].gt(0).all(), "non-positive price"
    assert model["sales_recorded"].gt(0).all(), "non-positive sales"
    assert brand["revenue_usd"].gt(0).all(), "non-positive revenue"
    return brand, model, shoe


def merge(brand, model, shoe):
    # 1) model-month + brand name
    df = model.copy()
    df["brand"] = df["model"].map(MODEL_TO_BRAND)
    assert df["brand"].notna().all(), "model missing from MODEL_TO_BRAND"

    # 2) brand-month stats (prefix so columns don't clash)
    b = brand.rename(columns={
        "sales_recorded": "brand_sales",
        "revenue_usd": "brand_revenue_usd",
        "mean_price_usd": "brand_mean_price_usd",
    })
    df = df.merge(b, on=["brand", "month"], how="left", validate="m:1")

    # 3) 12-month same-shoe change, model level (one row per model)
    m12 = (shoe[shoe["level"] == "model"]
           .drop(columns="level")
           .rename(columns={"name": "model"})
           .add_prefix("m12_")
           .rename(columns={"m12_model": "model"}))
    df = df.merge(m12, on="model", how="left", validate="m:1")

    # 4) 12-month same-shoe change, brand level (one row per brand)
    b12 = (shoe[shoe["level"] == "brand"]
           .drop(columns="level")
           .rename(columns={"name": "brand"})
           .add_prefix("b12_")
           .rename(columns={"b12_brand": "brand"}))
    df = df.merge(b12, on="brand", how="left", validate="m:1")

    # sanity: models are a subset of their brand, so model sales <= brand sales
    sums = df.groupby(["brand", "month"]).agg(
        s=("sales_recorded", "sum"), b=("brand_sales", "first"))
    assert (sums["s"] <= sums["b"]).all(), "model sales exceed brand sales"

    df["month"] = df["month"].dt.strftime("%Y-%m")
    cols = ["model", "brand", "month"] + [
        c for c in df.columns if c not in ("model", "brand", "month")]
    return df[cols].sort_values(["model", "month"]).reset_index(drop=True)


def make_features(df):
    """Lag features + next-month target (all built per model)."""
    df = df.sort_values(["model", "month"]).copy()
    g = df.groupby("model")
    df["lag_price"] = g["median_price_usd"].shift(1)
    df["lag_sales"] = g["sales_recorded"].shift(1)
    df["brand_lag_price"] = g["brand_mean_price_usd"].shift(1)
    df["target"] = g["median_price_usd"].shift(-1)
    df["month_num"] = pd.to_datetime(df["month"]).dt.month
    df["model_code"] = df["model"].astype("category").cat.codes
    df["brand_code"] = df["brand"].astype("category").cat.codes
    return df.dropna(subset=["lag_price", "target"])


def prepare_data():
    brand, model, shoe = clean(*load_raw())
    merged = merge(brand, model, shoe)
    merged.to_csv(MERGED_PATH, index=False)
    feats = make_features(merged)
    feats.to_csv(CLEAN_PATH, index=False)
    return merged, feats


if __name__ == "__main__":
    merged, feats = prepare_data()
    print(f"merged.csv: {merged.shape}")
    print(f"clean.csv : {feats.shape}")
