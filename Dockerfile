# Optimized Multi-Stage Production Dockerfile
# Stage 1: Build stage
FROM python:3.12-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        git \
        libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv and dependencies
COPY requirements.txt .
# Use torch-cpu to save space
RUN pip install --no-cache-dir uv && \
    uv pip install --system --no-cache -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Stage 2: Runtime stage
FROM python:3.12-slim

WORKDIR /app

# Install GDAL runtime libraries only
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gdal-bin \
        libgdal36 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed python packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application files
COPY alembic /app/alembic
COPY alembic.ini /app/
COPY src /app/src

# Environment
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8000
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Default command for development (can be overridden by docker-compose)
# Making K8s control the process with a service account
ENTRYPOINT [ "/app/entrypoint.sh" ]
