ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /app

# Service-specific lib overlays (shared/lib and predictor.py are in base)
COPY floods/lib /app/lib

ENV USECASE=floods

ENV MEANS="[1924.863, 2184.553, 2340.936, 3627.704, 3416.714, 2849.625]"
ENV STDS="[1201.092, 1219.943, 1397.225, 1447.836, 1471.002, 1365.30]"


# Expose port
EXPOSE 8001


CMD ["uvicorn", "predictor:app", "--host", "0.0.0.0", "--port", "8001" ]
