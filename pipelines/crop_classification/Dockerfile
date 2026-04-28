ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /app

# Copy requirements and install Python dependencies
COPY crop_classification/requirements.txt /app/crop_classification-requirements.txt

RUN uv pip install -r /app/crop_classification-requirements.txt --system

# Service-specific lib overlays (shared/lib and predictor.py are in base)
COPY crop_classification/lib /app/lib

ENV USECASE=crop_classification

ENV MEANS="[494.905781, 815.239594, 924.335066, 2968.881459, 2634.621962,1739.579917]"
ENV STDS="[284.925432, 357.84876, 575.566823, 896.601013, 951.900334, 921.407808]"
ENV NUM_CLASSES="13"

# Expose port
EXPOSE 8001

# Default command for development (can be overridden by docker-compose)
CMD ["uvicorn", "predictor:app", "--host", "0.0.0.0", "--port", "8001"]
