ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /app

# Copy requirements and install Python dependencies
COPY floods/requirements.txt /app/floods-requirements.txt

RUN uv pip install -r /app/floods-requirements.txt --system

RUN mkdir /app/lib
# Copy the rest of the application
COPY shared/lib /app/lib

COPY floods/lib /app/lib


COPY shared/predictor.py /app/

ENV USECASE=floods

ENV MEANS="[1924.863, 2184.553, 2340.936, 3627.704, 3416.714, 2849.625]"
ENV STDS="[1201.092, 1219.943, 1397.225, 1447.836, 1471.002, 1365.30]"


# Expose port
EXPOSE 8001

ENV PYTHONUNBUFFERED=1
ENV NUMBA_NUM_THREADS=1
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
##Intel MKL
ENV OPENBLAS_NUM_THREADS=1
ENV BLIS_NUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1
ENV VECLIB_MAXIMUM_THREADS=1


# Default command for development - using gunicorn with uvicorn worker and thread limits
CMD ["uvicorn", "predictor:app", "--host", "0.0.0.0", "--port", "8001" ]
#  "--loop", "uvloop", "--http", "h11", "--limit-concurrency", "1"