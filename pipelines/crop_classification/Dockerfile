ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /app

# Service-specific lib overlays (shared/lib, predictor.py, inference_worker.py are in base)
COPY crop_classification/lib /app/lib

ENV USECASE=crop_classification

ENV MEANS="[494.905781, 815.239594, 924.335066, 2968.881459, 2634.621962,1739.579917]"
ENV STDS="[284.925432, 357.84876, 575.566823, 896.601013, 951.900334, 921.407808]"
ENV NUM_CLASSES="13"

# Expose port
EXPOSE 8001

COPY crop_classification/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

CMD ["/app/entrypoint.sh"]
