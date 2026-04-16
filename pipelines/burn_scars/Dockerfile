ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /app

# Copy requirements and install Python dependencies
COPY burn_scars/requirements.txt /app/burn_scars-requirements.txt

RUN uv pip install -r /app/burn_scars-requirements.txt --system

RUN mkdir /app/lib
# Copy the rest of the application
COPY shared/lib /app/lib

COPY burn_scars/lib /app/lib

COPY shared/predictor.py /app/
COPY shared/inference_worker.py /app/inference_worker.py

ENV USECASE=burn_scars

ENV MEANS="[0.05350241946159966,0.07882975972154074,0.09625639684360336,0.2119436320676296,0.23597680308890565,0.17314308587875632]"
ENV STDS="[0.0308373261231399,0.03775894607078924,0.05499340234906741,0.0707235161610732,0.09191445860457703,0.08414614695771742]"

# Expose port
EXPOSE 8001

# Default command for development (can be overridden by docker-compose)
COPY burn_scars/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

CMD ["/app/entrypoint.sh"]
