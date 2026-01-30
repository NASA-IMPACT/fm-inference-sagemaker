# Use regular Python base image for local development
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y gcc g++ && apt-get clean git && apt-get install -y gdal-bin libgdal-dev

# Copy requirements and install Python dependencies
COPY requirements.txt /app/

RUN pip install uv && uv pip install -r /app/requirements.txt --system

COPY alembic /app/alembic

ADD alembic.ini /app

ARG DATABASE_URL

ENV DATABASE_URL=$DATABASE_URL

# RUN alembic -x dburl="${DATABASE_URL}" revision --autogenerate -m "create tables" && \
#     alembic -x dburl="${DATABASE_URL}" upgrade head

# Copy the rest of the application
COPY src /app/src

# Expose port
EXPOSE 8000

# Default command for development (can be overridden by docker-compose)
# Making K8s control the process with a service account
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
