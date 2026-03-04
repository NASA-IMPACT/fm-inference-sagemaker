ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# Install Python 3.12 using uv and create virtual environment
RUN uv python install 3.12
RUN uv venv /app/.venv --python 3.12

# Set environment to use the virtual environment
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Copy pyproject.toml for uv to install dependencies
COPY surya/pyproject.toml /app/pyproject.toml

# Install GDAL Python bindings matching system GDAL version
RUN uv pip install gdal==$(gdal-config --version)

# Verify GDAL installation
RUN python -c "from osgeo import gdal; print('GDAL version:', gdal.__version__)"

# Install Python dependencies using uv (includes transformers/tokenizers for Surya compatibility)
RUN uv pip install -e .

RUN git clone https://github.com/NASA-IMPACT/Surya.git /tmp/surya_source && \
    cd /tmp/surya_source && \
    uv pip install --no-cache --reinstall-package surya . && \
    rm -rf /tmp/surya_source

# Download Surya model from HuggingFace using git-lfs
RUN apt-get update && apt-get install -y git-lfs && git lfs install && \
    git clone https://huggingface.co/nasa-ibm-ai4science/Surya-1.0 /app/models/surya && \
    rm -rf /app/models/surya/.git && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set model path environment variables
ENV SURYA_CONFIG_PATH=/app/models/surya/config.yaml
ENV SURYA_SCALERS_PATH=/app/models/surya/scalers.yaml
ENV SURYA_WEIGHTS_PATH=/app/models/surya/surya.366m.v1.pt

RUN mkdir -p /app/lib

# Copy the library code
COPY surya/shared/lib /app/lib

# Copy the predictor
COPY surya/shared/predictor.py /app/

ENV USECASE=surya

# Expose port
EXPOSE 8001

# Default command for development (can be overridden by docker-compose)
CMD ["uvicorn", "predictor:app", "--host", "0.0.0.0", "--port", "8001"]
