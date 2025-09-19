# Use regular Python base image for local development
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y gcc g++ && apt-get clean git && apt-get install -y gdal-bin libgdal-dev

# Copy requirements and install Python dependencies
COPY requirements.txt /app/

RUN pip install uv && uv pip install -r /app/requirements.txt --system

# Dummy copy in case the migration needs to run 
COPY alembic /tmp

# Copy the db_migration
COPY db_migration.sh /db_migration.sh
RUN bash /db_migration.sh

# Copy the rest of the application
COPY . /app/

# Expose port
EXPOSE 8000

# Default command for development (can be overridden by docker-compose)
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "10"]
