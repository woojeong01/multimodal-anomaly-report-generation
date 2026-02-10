# Optional (not required for Step 1)
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -U pip && pip install -r requirements.txt

COPY . .
ENV RUNTIME_CFG=configs/anomaly.yaml
EXPOSE 8000
CMD ["uvicorn", "apps.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
