import json
import os

import joblib
import mlflow
import pandas as pd
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from src.validate import validate

PROCESSED_PATH = "data/processed/clean.csv"
MODEL_PATH = "models/model.pkl"
LOOKUP_PATH = "models/lookup.json"

FEATURES = [
    "median_price_usd",      # current month price
    "lag_price",             # previous month price
    "sales_recorded",        # current month sales
    "brand_mean_price_usd",  # brand average price
    "month_num",
    "price_change",
    "price_change_pct",
    #"model_code",
    #"brand_code",
]
TEST_MONTHS = 3  # hold out the last 3 months for testing


def train():
    df = validate(path=PROCESSED_PATH).sort_values("month")
    df["price_change"] = df["median_price_usd"] - df["lag_price"]
    df["price_change_pct"] = df["price_change"] / df["lag_price"]

    # Time-based split: earlier months for training, last 3 months for testing
    test_start = sorted(df["month"].unique())[-TEST_MONTHS]
    train_df = df[df["month"] < test_start]
    test_df = df[df["month"] >= test_start]

    X_train, y_train = train_df[FEATURES], train_df["target"]
    X_test, y_test = test_df[FEATURES], test_df["target"]

    y_train_delta = y_train - train_df["median_price_usd"]
    y_test_delta = y_test - test_df["median_price_usd"]

    # Baseline for comparison: "next month price = current month price"
    baseline_mae = mean_absolute_error(y_test, test_df["median_price_usd"])

    params = {
    "n_estimators": 50,
    "max_depth": 2,
    "learning_rate": 0.05,
    "min_child_weight": 5,
    "objective": "reg:absoluteerror",
    }
    
    mlflow.set_experiment("sneaker-resale")
    with mlflow.start_run():
        model = XGBRegressor(**params)
        model.fit(X_train, y_train_delta)

        pred = test_df["median_price_usd"] + model.predict(X_test)
        mae = mean_absolute_error(y_test, pred)

        mlflow.log_params(params)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("baseline_mae", baseline_mae)

    print(f"Train rows: {len(train_df)}, Test rows: {len(test_df)}")
    print(f"Baseline MAE: {baseline_mae:.2f} USD")
    print(f"Model MAE   : {mae:.2f} USD")

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    # Lookup for the API: model name -> model_code and brand_code
    lookup = (
        df[["model", "model_code", "brand", "brand_code"]]
        .drop_duplicates()
        .set_index("model")
        .to_dict("index")
    )
    with open(LOOKUP_PATH, "w") as f:
        json.dump(lookup, f, indent=2)
    print("Saved: models/model.pkl, models/lookup.json")


if __name__ == "__main__":
    train()