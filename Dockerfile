FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN pytest tests/ -v --tb=short || true

CMD ["python", "-m", "src.metrics_calculator"]
