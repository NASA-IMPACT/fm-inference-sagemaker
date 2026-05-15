#!/usr/bin/env bash
set -e

timeout="${MODEL_SERVER_TIMEOUT:-60}"

ln -sf /dev/stdout /var/log/nginx/access.log
ln -sf /dev/stderr /var/log/nginx/error.log

nginx -c /opt/program/nginx.conf &
nginx_pid=$!

gunicorn \
    --timeout "${timeout}" \
    -b unix:/tmp/gunicorn.sock \
    -w 1 \
    --worker-class uvicorn.workers.UvicornWorker \
    predictor:app &
gunicorn_pid=$!

trap 'kill -QUIT "${nginx_pid}" 2>/dev/null; kill -TERM "${gunicorn_pid}" 2>/dev/null; exit 0' TERM INT

wait -n "${nginx_pid}" "${gunicorn_pid}"
kill -QUIT "${nginx_pid}" 2>/dev/null || true
kill -TERM "${gunicorn_pid}" 2>/dev/null || true
