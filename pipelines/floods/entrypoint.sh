#!/bin/sh
set -e

# Start FastAPI in background
uvicorn predictor:app --host 0.0.0.0 --port 8001 &
API_PID=$!

# Trap signals and forward to FastAPI
trap "echo 'Stopping FastAPI...'; kill -TERM $API_PID; wait $API_PID; exit 0" INT TERM

# Start Temporal worker in foreground (PID 1)
python3 -m inference_worker
