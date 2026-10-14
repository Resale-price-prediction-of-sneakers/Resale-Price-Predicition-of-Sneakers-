FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock .python-version ./ 
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
COPY models ./models

EXPOSE 8000
CMD ["uv", "run", "--no-sync", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]

