ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /app

# Copy pyproject.toml for uv to install dependencies
COPY surya/pyproject.toml /app/pyproject.toml

# Install Python dependencies using uv (includes transformers/tokenizers for Surya compatibility)
RUN uv pip install -e .

# Install Surya library from source
RUN git clone https://github.com/NASA-IMPACT/Surya.git /tmp/surya_source && \
    cd /tmp/surya_source && \
    uv pip install --reinstall-package surya . && \
    rm -rf /tmp/surya_source

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
