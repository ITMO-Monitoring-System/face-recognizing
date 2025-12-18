# syntax=docker/dockerfile:1.6
FROM python:3.11-slim-bookworm
WORKDIR /app

RUN apt-get update -o Acquire::ForceIPv4=true -o Acquire::Retries=5 -o Acquire::http::Timeout=30 && \
    apt-get install -y --no-install-recommends \
      libgl1 libglib2.0-0 build-essential g++ && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -U pip setuptools wheel && \
    pip install -r requirements.txt

COPY . .
CMD ["python", "worker.py"]
