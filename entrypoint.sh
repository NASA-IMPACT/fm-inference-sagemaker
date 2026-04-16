#!/bin/sh
set -e

# Start FastAPI in background
uvicorn src.main:app --host 0.0.0.0 --port 8000 --backlog  20 --workers 1 --timeout-keep-alive 120 &
API_PID=$!

# Trap signals and forward to FastAPI
trap "echo 'Stopping FastAPI...'; kill -TERM $API_PID; wait $API_PID; exit 0" INT TERM

# Start Temporal worker in foreground (PID 1)
python3 -m src.workflows.worker
