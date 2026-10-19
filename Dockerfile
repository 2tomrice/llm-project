FROM python:3.13-slim

WORKDIR /app
COPY requirements-api.txt .

RUN pip install --no-cache-dir -r requirements-api.txt
COPY main.py .
COPY model_service.py .
COPY outputs/cs-tutor-lora-v2/checkpoint-141/adapter_config.json ./model_adapter/
COPY outputs/cs-tutor-lora-v2/checkpoint-141/adapter_model.safetensors ./model_adapter/

EXPOSE 8080

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]