ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /app

# Service-specific lib overlays (shared/lib and predictor.py are in base)
COPY burn_scars/lib /app/lib

ENV USECASE=burn_scars

ENV MEANS="[0.05350241946159966,0.07882975972154074,0.09625639684360336,0.2119436320676296,0.23597680308890565,0.17314308587875632]"
ENV STDS="[0.0308373261231399,0.03775894607078924,0.05499340234906741,0.0707235161610732,0.09191445860457703,0.08414614695771742]"

# Expose port
EXPOSE 8001

# Default command for development (can be overridden by docker-compose)
CMD ["uvicorn", "predictor:app", "--host", "0.0.0.0", "--port", "8001"]
