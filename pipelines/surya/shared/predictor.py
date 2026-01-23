import boto3
import gc
import GPUtil
import httpx
import os
import re
import torch

from botocore import UNSIGNED
from botocore.config import Config
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from fastapi import FastAPI, Request, APIRouter, status, Body, Depends, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from glob import glob
from pydantic import BaseModel
from typing import Optional, Tuple, List

from lib.consts import (
    CHANNELS, DOWNLOAD_FOLDER, OUTPUT_DIR, EMAIL, FITS_DIR,
    SURYA_CONFIG_PATH, SURYA_SCALERS_PATH, SURYA_WEIGHTS_PATH,
    SURYA_S3_BUCKET
)
from lib.rollout_infer import Infer
from lib.downloader import Downloader
from lib.data_process import DataProcess

# Output directories
PREDICTION_FOLDER = f"{DOWNLOAD_FOLDER}/predictions"
os.makedirs(PREDICTION_FOLDER, exist_ok=True)

# Create FastAPI apps
app = FastAPI(
    docs_url=None,
    redoc_url=None,
)

v1_api = FastAPI(
    title="Surya Predictor API",
    description="Solar Dynamics Observatory (SDO) Rollout Forecasting API",
    version="1.0.0"
)

# --- Security Configuration ---
API_KEY_VALIDATION_URL = os.getenv("API_KEY_VALIDATION_URL", "https://dev.fm.dsig.net/api/validate")
api_key_header = APIKeyHeader(name="x-api-key")

# --- Model Configuration ---
DATA_DIR = os.getenv("SURYA_DATA_DIR", f"{DOWNLOAD_FOLDER}/sdo_data")
WIND_DATA_DIR = os.getenv("SURYA_WIND_DATA_DIR", f"{DOWNLOAD_FOLDER}/wind_data")
NETCDF_DIR = os.path.join(FITS_DIR, "processed_netcdf")

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(NETCDF_DIR, exist_ok=True)


def download_single_timestamp(args):
    """Download data for a single timestamp. Must be top-level for ProcessPoolExecutor."""
    ts, cadence_minutes, email, fits_dir, wind_data_dir = args
    ts_key = ts.strftime("%Y%m%d_%H%M%S")
    print(f"Downloading data for timestamp: {ts_key}")

    downloader = Downloader(email=email, output_dir=fits_dir, wind_data_dir=wind_data_dir)
    download_results = downloader.download(ts, ts, cadence=f"{cadence_minutes}m")

    raw_fits_dir = os.path.join(fits_dir, "raw_fits")

    # Organize files for this timestamp
    aia_files = []
    hmi_files = {
        'magnetogram': None,
        'doppler': None,
        'field': None,
        'inclination': None,
        'azimuth': None,
        'disambig': None,
    }

    for task_name, result in download_results.items():
        all_files = result.get('downloaded', []) + result.get('skipped', [])
        for fname in all_files:
            fpath = os.path.join(raw_fits_dir, fname)
            if 'aia' in fname.lower():
                aia_files.append(fpath)
            elif 'hmi.m_720s' in fname:
                hmi_files['magnetogram'] = fpath
            elif 'hmi.v_720s' in fname:
                hmi_files['doppler'] = fpath
            elif 'hmi.b_720s' in fname:
                if '.field.' in fname:
                    hmi_files['field'] = fpath
                elif '.inclination.' in fname:
                    hmi_files['inclination'] = fpath
                elif '.azimuth.' in fname:
                    hmi_files['azimuth'] = fpath
                elif '.disambig.' in fname:
                    hmi_files['disambig'] = fpath

    return ts_key, {'aia_files': aia_files, 'hmi_files': hmi_files}


def assign_available_gpus():
    """Assign available GPUs to the current process using GPUtil (least memory usage)."""
    try:
        free_gpus = GPUtil.getAvailable(order='memory', limit=1)
        if os.environ.get("GPU_ID"):
            free_gpus = GPUtil.getAvailable(order='memory', limit=8)
            available_gpus = [int(gpu_id) for gpu_id in os.environ["GPU_ID"].split(",")]
            free_gpus = [gpu_id for gpu_id in free_gpus if gpu_id in available_gpus]
        if free_gpus:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(free_gpus[0])
            print(f"Assigned GPU: {free_gpus[0]}")
        else:
            print("No free GPUs found.")
    except Exception as e:
        print(f"Could not assign GPU automatically: {e}")


def load_model():
    """Load and initialize the Surya model."""
    infer = Infer(
        config_path=SURYA_CONFIG_PATH,
        scalers_path=SURYA_SCALERS_PATH,
        weights_path=SURYA_WEIGHTS_PATH,
        data_dir=NETCDF_DIR,
        results_dir=OUTPUT_DIR
    )
    infer.load_model()
    return infer

def download_from_s3(ts: datetime) -> Optional[Tuple[str, str]]:
    """
    Attempt to download NetCDF data file from S3 for the given timestamp.

    Args:
        ts: Timestamp to download
    Returns:
        Tuple of (ts_key, nc_file_path) if successful, else None
    """
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    ts_key = ts.strftime("%Y%m%d_%H%M")
    year, month = ts.strftime("%Y"), ts.strftime("%m")
    s3_prefix = f"{year}/{month}"
    s3_key = f"{s3_prefix}/{ts_key}.nc"
    nc_file_path = os.path.join(NETCDF_DIR, f"{ts_key}00.nc")
    try:
        s3.download_file(SURYA_S3_BUCKET, s3_key, nc_file_path)
        print(f"Downloaded NetCDF file from S3: {s3_key}")
        return ts, nc_file_path
    except Exception as e:
        print(f"S3 download failed for {s3_key}: {e}")
        return None, None


def parallel_download_from_s3(download_args: List[Tuple[datetime, int, str, str, str]]) -> List[datetime]:
    """
    Download NetCDF files from S3 in parallel.

    Args:
        download_args: List of tuples containing (ts, cadence_minutes, email, fits_dir, wind_data_dir)
    """
    nc_files = []
    downloaded_timestamps = []

    max_workers = min(len(download_args), 12)  # Increased from 4 to 12 for faster downloads
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_from_s3, args[0]): args[0] for args in download_args}

        for future in as_completed(futures):
            ts = futures[future]
            ts, nc_file = future.result()
            if nc_file:
                nc_files.append(nc_file)
                print(f"Completed download for timestamp: {nc_file}")
                downloaded_timestamps.append(ts)
            else:
                print(f"S3 download failed for timestamp {ts.strftime('%Y%m%d_%H%M%S')}, will attempt direct download.")
    return downloaded_timestamps, nc_files


def parallel_download_and_preprocess(download_args: List[Tuple[datetime, int, str, str, str]]) -> List[str]:
    """
    Download and preprocess NetCDF files in parallel.

    Args:
        download_args: List of tuples containing (ts, cadence_minutes, email, fits_dir, wind_data_dir)
    """
    nc_files = []
    # Download all timestamps in parallel using ProcessPoolExecutor
    files_by_timestamp = {}

    max_workers = min(len(download_args), 12)  # Increased from 4 to 12 for faster downloads
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_single_timestamp, args): args[0] for args in download_args}

        for future in as_completed(futures):
            ts = futures[future]
            try:
                ts_key, ts_data = future.result()
                files_by_timestamp[ts_key] = ts_data
                print(f"Completed download for timestamp: {ts_key}")
            except Exception as e:
                print(f"Error downloading timestamp {ts.strftime('%Y%m%d_%H%M%S')}: {e}")

    # Get sorted list of available timestamps
    sorted_available_timestamps = sorted(files_by_timestamp.keys())
    print(f"Downloaded files for {len(sorted_available_timestamps)} timestamps: {sorted_available_timestamps}")

    # Process each timestamp to NetCDF
    for ts_key in sorted_available_timestamps:
        ts_data = files_by_timestamp[ts_key]
        aia_files = sorted(ts_data['aia_files'])
        hmi_files = ts_data['hmi_files']
        processor = DataProcess(aia_files, hmi_files)
        filename = processor.process_timestamp(FITS_DIR, ts_key)
        nc_files.append(filename)

    return nc_files


def ensure_data_available(start_time: datetime, cadence_minutes: int = 12, num_frames: int = 1) -> Tuple[List[str], List[str]]:
    """
    Ensure NetCDF data files are available for the requested timestamp.
    Downloads and preprocesses if needed.

    Args:
        start_time: Requested start datetime
        cadence_minutes: Cadence between frames
        num_frames: Number of frames to download

    Returns:
        Tuple of (nc_files, available_times) where:
            - nc_files: List of processed NetCDF file paths
            - available_times: List of timestamp keys (YYYYMMDD_HHMMSS format)
    """
    # TODO: Add caching/early return logic for existing NetCDF files

    # Download and process new data
    if not EMAIL:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="SDO_EMAIL not configured. Cannot download data."
        )

    print(f"No NetCDF files found. Downloading data for {start_time}...")

    # Generate list of timestamps to download
    total_timestamps = [
        start_time + timedelta(minutes=cadence_minutes * i)
        for i in range(-1, num_frames + 1)
    ]
    nc_files = []
    timestamps_to_download = []

    for ts in total_timestamps:
        ts_key = ts.strftime("%Y%m%d_%H%M%S")
        nc_file = os.path.join(NETCDF_DIR, f"{ts_key}.nc")
        if not os.path.exists(nc_file):
            timestamps_to_download.append(ts)
        else:
            nc_files.append(nc_file)
            print(f"NetCDF file already exists for timestamp {ts_key}, skipping download.")

    if not timestamps_to_download:
        print("All requested NetCDF files are already available.")
        return nc_files

    # Prepare arguments for parallel download
    download_args = [
        (ts, cadence_minutes, EMAIL, FITS_DIR, WIND_DATA_DIR)
        for ts in timestamps_to_download
    ]

    # Try Download data from S3 first
    downloaded_timestamps, s3_ncfiles = parallel_download_from_s3(download_args)
    nc_files.extend(s3_ncfiles)

    timestamps_to_download = set(timestamps_to_download) - set(downloaded_timestamps)

    if timestamps_to_download:
        download_args = [
            (ts, cadence_minutes, EMAIL, FITS_DIR, WIND_DATA_DIR)
            for ts in timestamps_to_download
        ]
        prepocessed_timestamps = parallel_download_and_preprocess(download_args)
        nc_files.extend(prepocessed_timestamps)

    sorted_nc_files = sorted(nc_files)

    return sorted_nc_files


# Assign GPU before loading the model
assign_available_gpus()

# Load model at startup
MODEL = load_model()

public_router = APIRouter()

# --- Request/Response Models ---
class RolloutRequest(BaseModel):
    selected_datetime: str  # ISO format: "2024-06-10T12:00:00"
    cadence_in_minutes: Optional[int] = 60
    num_frames: Optional[int] = 5


class ChannelScore(BaseModel):
    timestamp: str
    score: dict[str, float]


class RolloutStep(BaseModel):
    timestamp: str
    tiles_endpoint: str
    metadata_endpoint: str


class RolloutQuery(BaseModel):
    selected_datetime: str
    cadence_in_minutes: int
    num_frames: int


class RolloutData(BaseModel):
    query: RolloutQuery
    steps: list[RolloutStep]
    correlation: list[ChannelScore]


class RolloutResponse(BaseModel):
    surya: dict


# --- Inference Logic ---
def run_rollout_inference(selected_datetime: str, cadence_minutes: int, num_frames: int):
    """
    Run rollout inference for the given datetime and parameters.

    Args:
        selected_datetime: ISO format datetime string
        cadence_minutes: Time between forecast steps
        num_frames: Number of forecast frames to generate

    Returns:
        Formatted API response dictionary
    """
    # Parse the selected datetime
    try:
        start_time = datetime.fromisoformat(selected_datetime)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid datetime format: {e}"
        )

    # Ensure data is available (downloads if needed)
    nc_files = ensure_data_available(start_time, cadence_minutes, num_frames)

    # Find the two closest files to the selected datetime
    first_file, second_file = nc_files[0], nc_files[1]
    print(f"Using files for inference: {first_file}, {second_file}")
    print(nc_files, 'NC FILES AND AVAILABLE TIMESTAMPS')
    # Run inference
    try:
        results = MODEL.run_inference(
            first_step_file=first_file,
            second_step_file=second_file,
            steps=num_frames,
            cadence_minutes=cadence_minutes
        )
    except Exception as e:
        print(f"Inference error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {str(e)}"
        )

    # Build forecast config for process_results
    forecast_config = {
        'start_time': start_time,
        'cadence_minutes': cadence_minutes,
        'max_steps': num_frames
    }

    # Process results into API format
    response = MODEL.process_results(results, forecast_config)

    # Cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return response


# --- API Endpoints ---
@public_router.post('/invocations')
async def rollout_forecast(request: RolloutRequest = Body(...)):
    """
    Run rollout forecasting for solar imagery.

    Returns forecast tiles endpoints and correlation scores for each timestep.
    """
    print(f"Received rollout request: datetime={request.selected_datetime}, "
          f"cadence={request.cadence_in_minutes}min, frames={request.num_frames}")

    response = run_rollout_inference(
        selected_datetime=request.selected_datetime,
        cadence_minutes=request.cadence_in_minutes,
        num_frames=request.num_frames
    )

    return JSONResponse(content=jsonable_encoder(response))


@public_router.get('/channels')
async def get_channels():
    """Return the list of available SDO channels."""
    return JSONResponse(content=jsonable_encoder({
        "channels": CHANNELS,
        "count": len(CHANNELS)
    }))


@public_router.get('/ping')
async def ping(request: Request):
    """Health check endpoint."""
    return {"successCode": 200, "message": "pong"}


@public_router.get("/health")
async def health():
    """Health status endpoint."""
    return {
        "successCode": 200,
        "status": "healthy",
        "model_loaded": MODEL is not None,
        "gpu_available": torch.cuda.is_available()
    }


# Include routers
v1_api.include_router(public_router)
app.mount("/api/v1", v1_api)
