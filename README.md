# Resale-Price-Prediction-of-Sneakers

![CI](https://github.com/Resale-Price-Prediction-of-Sneakers/Resale-Price-Prediction-of-Sneakers-/ons/workflows/ci-yml/badge.svg)

MLOps pipeline for real-time sneaker resale price prediction (GOAT + StockX data): data processing, model training, a FastAPI prediction service, Docker packaging and CI.

## Project structure
```
src/        data.py (features), train.py (training), api.py (FastAPI service)
tests/      pytest tests for data and API
models/     model.pkl, lookup.json
data/       raw and processed CSV files
Dockerfile  container image for the API
.github/    CI workflow (tests + Docker build)
```

## Setup
Requires [uv](https://docs.astral.sh/uv/) and Python 3.11.
```
uv sync
```

## Train
```
uv run python -m src.train
```

The model is evaluated with MAE against a baseline.

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

Unknown model names return `404`; invalid input returns `422`.

## Tests
```
uv run pytest
```

## CI
GitHub Actions runs the tests and builds the Docker image on every push and pull request.
