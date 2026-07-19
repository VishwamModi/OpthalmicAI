FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 libgl1 libglib2.0-0 libxcb1 \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements-cpu.txt /app/backend/requirements-cpu.txt
RUN python -m pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        torch torchvision \
    && python -m pip install --no-cache-dir -r /app/backend/requirements-cpu.txt

COPY backend /app/backend
COPY Models/OphthalmicAI_Final_EffNetB3.pt /app/Models/OphthalmicAI_Final_EffNetB3.pt

EXPOSE 7860

CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT}"]
