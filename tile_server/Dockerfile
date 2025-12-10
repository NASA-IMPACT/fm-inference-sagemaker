ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /app

# Copy requirements and install Python dependencies
COPY tile_server/requirements.txt /app/tile_server-requirements.txt

RUN uv pip install -r /app/tile_server-requirements.txt --system


COPY tile_server/tile_server.py /app/

# Expose port
EXPOSE 8090

# Default command for development (can be overridden by docker-compose)
CMD ["uvicorn", "tile_server:app", "--host", "0.0.0.0", "--port", "8090"]
