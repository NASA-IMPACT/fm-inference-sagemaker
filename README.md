# fm-inference-sagemaker

Inference service for NASA IMPACT foundation-model use cases (flood segmentation by default). The service is a FastAPI application that exposes SageMaker-compatible `/ping` and `/invocations` endpoints. It can be run two ways:

- **Docker / SageMaker**: container with nginx in front of gunicorn + uvicorn, listening on port 8080. nginx is only used in this mode.
- **Local development**: a `uv` virtual environment on Python 3.11 running uvicorn directly.

## Repository layout

- `Dockerfile` — CUDA 12.1 / Ubuntu 22.04 image with Python 3.11, GDAL, and the inference stack.
- `requirements.txt` — Python dependencies (PyTorch, terratorch, rasterio, rio-cogeo, FastAPI, etc.).
- `build_and_push.sh` — Builds the image and pushes it to ECR (`us-west-2`).
- `code/entrypoint.sh` — Container entrypoint; starts nginx and gunicorn.
- `code/nginx.conf` — Reverse proxy config used inside the container only.
- `code/wsgi.py` — WSGI shim.
- `code/predictor.py` — FastAPI app; loads the model on startup and implements inference.
- `code/lib/`
  - `consts.py` — Constants and env-driven config (bucket, layers, crop size).
  - `downloader.py` — Downloads HLS tiles for a given date and bounding box.
  - `infer.py` — Segmentation inference.
  - `infer_generation.py` — Tiled generation inference path.
  - `post_process.py` — Contour extraction, GeoJSON conversion, intersection cleanup.
  - `utils.py` — AWS session helpers.

## Endpoints

- `GET /ping` — Health check. Returns `{"successCode": 200, "message": "pong"}`.
- `POST /invocations` — Runs inference. JSON body.

### Segmentation request

```json
{
  "date": "YYYY-MM-DD",
  "bounding_box": [minx, miny, maxx, maxy],
  "terramind": false,
  "file_urls": []
}
```

When `terramind` is `true`, the listed `file_urls` (S3 paths) are used as inputs instead of fetching HLS tiles by date and bounding box.

Response:

```json
{
  "<usecase>": {
    "s3_link": "s3://<bucket>/predictions/<timestamp>-predictions.tif",
    "predictions": { "type": "FeatureCollection", "features": [ ... ] }
  }
}
```

### Generation request

```json
{
  "generation": true,
  "input_file": "<path-or-s3-uri>",
  "reduce": true
}
```

Returns the tiled generation output as JSON.

## Configuration

Set via environment variables (defaults are baked into the `Dockerfile` and can be overridden at runtime).

| Variable | Required | Description |
| --- | --- | --- |
| `BUCKET_NAME` | yes | S3 bucket containing the config, checkpoint, and used for prediction uploads. |
| `S3_CONFIG_FILENAME` | yes | S3 path to the model config YAML. |
| `CHECKPOINT_FILENAME` | yes | S3 path to the model checkpoint. |
| `USECASE` | yes | Use case key (e.g. `flood`). Used as the model id in responses. |
| `MODEL_SERVER_TIMEOUT` | no | Gunicorn worker timeout in seconds (container only). Defaults to `150`. |

On startup, `predictor.py` downloads the config and checkpoint from S3 (assumed-role session via `lib/utils.py`) into `./config` and `./models`, then constructs an `Infer` instance.

## Local development (uv + Python 3.11)

Requires Python 3.11, GDAL system libraries, and AWS credentials with access to the configured bucket.

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt

export BUCKET_NAME=<bucket>
export S3_CONFIG_FILENAME=<s3-path-to-config.yaml>
export CHECKPOINT_FILENAME=<s3-path-to-checkpoint>
export USECASE=flood

cd code
uvicorn predictor:app --host 0.0.0.0 --port 8080
```

Then:

```bash
curl http://localhost:8080/ping
curl -X POST http://localhost:8080/invocations \
  -H 'Content-Type: application/json' \
  -d '{"date":"2024-08-01","bounding_box":[-90.2,29.9,-89.8,30.2]}'
```

A CUDA-capable GPU is required at inference time (`torch.cuda.synchronize` / `torch.cuda.empty_cache` are called on the inference path).

## Docker

### Build

```bash
docker build . -f Dockerfile --platform linux/amd64 -t fm_inference
```

### Push to ECR

`build_and_push.sh` builds for `linux/amd64`, logs in to ECR in `us-west-2`, and pushes `fm_inference:latest`. It expects `AWS_ACCOUNT_ID` in the environment.

```bash
export AWS_ACCOUNT_ID=<your-account-id>
./build_and_push.sh
```

### Run

```bash
docker run --rm -p 8080:8080 \
  -e BUCKET_NAME=<bucket> \
  -e S3_CONFIG_FILENAME=<s3-path-to-config.yaml> \
  -e CHECKPOINT_FILENAME=<s3-path-to-checkpoint> \
  -e USECASE=flood \
  -e AWS_ACCESS_KEY_ID=... \
  -e AWS_SECRET_ACCESS_KEY=... \
  -e AWS_SESSION_TOKEN=... \
  --gpus all \
  fm_inference
```

In this mode nginx terminates HTTP on port 8080 and proxies `/ping` and `/invocations` to gunicorn over a Unix socket.

## Notes

- Segmentation inputs are HLS tiles (`HLSS30`, `HLSL30`) downloaded for the requested date and bounding box.
- Predictions are merged into a mosaic, written as a Cloud-Optimized GeoTIFF, and uploaded to `s3://$BUCKET_NAME/predictions/`.
- GeoJSON output is clipped to the requested bounding box before being returned.
