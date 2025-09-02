FROM nvidia/cuda:12.1.1-base-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y gcc g++ && \
    apt-get clean git && \
    apt-get install -y gdal-bin libgdal-dev && \
    apt-get install -y ffmpeg libsm6 libxext6 libgl1 && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY floods/requirements.txt /app/

RUN pip install uv && uv pip install -r /app/requirements.txt --system

RUN mkdir /app/lib
# Copy the rest of the application
COPY shared/lib /app/lib

COPY shared/predictor.py /app/

# Expose port
EXPOSE 8001

# Default command for development (can be overridden by docker-compose)
CMD ["uvicorn", "predictor:app", "--host", "0.0.0.0", "--port", "8001"]
