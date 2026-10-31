# Resale-Price-Prediction-of-Sneakers

![CI](https://github.com/Resale-price-prediction-of-sneakers/Resale-Price-Prediction-of-Sneakers-/actions/workflows/ci.yml/badge.svg)

MLOps pipeline for real-time sneaker resale price prediction (GOAT + StockX data): data processing, model training, a FastAPI prediction service, Docker packaging and CI.

## Project structure
```
src/        data.py (features), train.py (training), api.py (FastAPI service)
tests/      pytest tests for data and API
models/     model.pkl, lookup.json
data/       raw (tracked by DVC) and processed CSV files
grafana/    dashboard JSON export
Dockerfile  container image for the API
docker-compose.yml, prometheus.yml   API + Prometheus + Grafana
.github/    CI workflow (tests + Docker build)
```

## Data

Monthly sneaker resale prices from GOAT and StockX:
`sneakerpulse_brand_monthly.csv`, `sneakerpulse_model_monthly_price.csv`,`sneakerpulse_same_shoe_12m.csv` (in `data/raw/`). The cleaned and merged table used for training is `data/processed/clean.csv`.

Features: current median price, previous month price (lag), sales recorded, brand mean price, month number, price change and price change %.
Target: next-month resale price.

Raw data is versioned with DVC (`data/raw.dvc`). No DVC remote is configured yet.

## Setup
Requires [uv](https://docs.astral.sh/uv/) and Python 3.11.
```
uv sync
```

## Train
```
uv run python -m src.train
```
An XGBoost model predicts the price change for next month. The last 3 months are held out for testing, and the baseline assumes next month’s price equals this month’s price. Runs are logged with MLflow (experiment `sneaker-resale`):

    uv run mlflow ui —backend-store-uri sqlite:///mlflow.db

Results are in the Results section below.

## Run the API
Locally:
```
uv run uvicorn src.api:app --reload
```

With Docker:
```
docker build -t mlops-lab .
docker run -p 8000:8000 mlops-lab
```
Open http://localhost:8000/docs for the interactive docs.

### Example request
`POST /predict`

```json
{
    "model": "Air Force 1",
    "median_price_usd": 110,
    "lag_price": 100,
    "sales_recorded": 20,
    "brand_mean_price_usd": 115,
    "month_num": 2
}
```

Response:
```json
{
  "model": "Air Force 1",
  "predicted_next_month_price": 108.45,
  "predicted_change": -1.55
}
```
`predicted_change` = `predicted_next_month_price` - `mediam_price_usd`.

Unknown model names return `404`; invalid input returns `422`.

## Tests
```
uv run pytest
```
One test needs the raw data and is skipped when it is not available (for example in CI).

## CI
GitHub Actions runs the tests and builds the Docker image on every push and pull request.

## Monitoring

Prometheus and Grafana run alongside the API with Docker Compose.

```
docker compose up -d
```

| Service | URL |
|---|---|
| API docs | http://localhost:8000/docs |
| Metrics | http://localhost:8000/metrics |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 |

The Grafana dashboard has four panels: API Up, API Request Rate, Average Predicted Price and Memory Usage. 
Its JSON export is in `grafana/dashboards/api-dashboard.json`. 
Grafana data persists in a Docker volume, so do not run docker compose down -v.

## Results

Time-split evaluation: The last 3 months are held out for testing (442 train rows, 78 test rows). The baseline predicts that next month’s price equals the current month’s price.

| Model | MAE (USD) |
|---|---|
| Baseline | 5.29 |
| XGBoost | 5.27 |

The model is only slightly better than the baseline. The dataset is small, so the focus of this project is the MLOps pipeline (tracking, API, Docker, CI, monitoring) rather than model accuracy. 

## Future work

More data and features, hyperparameter tuning, data and prediction drift monitoring, and a DVC remote so CI can pull the raw data.

