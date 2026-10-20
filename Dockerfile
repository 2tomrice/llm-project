FROM python:3.13-slim

WORKDIR /app
COPY requirements-api.txt .

RUN pip install --no-cache-dir -r requirements-api.txt
COPY app ./app
ENV ADAPTER_PATH=/app/model_adapter
COPY outputs/cs-tutor-lora-v2/checkpoint-141/adapter_config.json ./model_adapter/
COPY outputs/cs-tutor-lora-v2/checkpoint-141/adapter_model.safetensors ./model_adapter/

EXPOSE 8080

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}"]