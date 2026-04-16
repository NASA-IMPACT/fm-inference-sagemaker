import asyncio
import geopandas as gpd
import httpx
import importlib
import inflection
import inspect
import json
import numpy as np
import os
import rasterio
import time
import threading

from fastapi import FastAPI, Request, APIRouter, status, Body, Depends, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader

from lib.consts import (
    BUCKET_NAME,
    CONFIG_PATH,
    MODEL_WEIGHT_PATH,
    USECASE,
    DOWNLOAD_FOLDER,
    NUM_CLASSES,
)
from lib.data_preparer import DataPreparer
from lib.post_process import PostProcess
from lib.utils import (
    get_boto3_session,
    upload_cog_to_s3,
    async_upload_many_to_s3,
)

from pydantic import BaseModel

from rasterio.io import MemoryFile
from rasterio.mask import mask
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject, Resampling

from shapely.geometry import shape, box

from typing import Optional
from concurrent.futures import ThreadPoolExecutor

PREDICTION_FOLDER = f"{DOWNLOAD_FOLDER}/predictions"
os.makedirs(PREDICTION_FOLDER, exist_ok=True)
MODEL = {}
model_lock = threading.Lock()


# This will be served by the FastAPI as a container
# Re-enable docs to see the authorization feature
# Create without docs
app = FastAPI(docs_url=None, redoc_url=None)

# Todo Provide a better title
v1_api = FastAPI(
    title="Predictor API for Floods",
    description="Predictor API Floods (MVP)",
    version="0.0.1",
)

# --- Start of Modified Security Block ---

# This is the URL of your external validation service.
API_KEY_VALIDATION_URL = os.getenv(
    "API_KEY_VALIDATION_URL", "https://dev.fm.dsig.net/api/validate"
)

# Fixed: Remove auto_error=False to make it work with FastAPI's authorization UI
api_key_header = APIKeyHeader(name="x-api-key")


def assign_available_gpus():
    """Round Robin GPU assignment"""
    try:
        gpu_ids = (
            os.environ.get("GPU_ID").split(",") if os.environ.get("GPU_ID") else None
        )
        if gpu_ids:
            pod_id = int(os.environ.get("POD_INDEX", 0))
            selected_gpu_id = gpu_ids[pod_id % len(gpu_ids)]
            os.environ["CUDA_VISIBLE_DEVICES"] = str(selected_gpu_id)
            print(f"Assigned GPU {selected_gpu_id} to this instance (pod_id={pod_id})")
        else:
            print("No GPU IDs assigned; Assigning all GPUs available.")
    except Exception as e:
        print(f"Could not assign GPU automatically: {e}")


def download_from_s3(s3_path, download_path="config"):
    download_path = f"{DOWNLOAD_FOLDER}/{download_path}"
    session = get_boto3_session()
    s3_connection = session.resource("s3")
    bucket = s3_connection.Bucket(BUCKET_NAME)
    filename = s3_path.split("/")[-1]
    file_path = f"{download_path}/{filename}"
    if not os.path.exists(file_path):
        os.makedirs(download_path, exist_ok=True)
        os.makedirs("predictions", exist_ok=True)
        bucket.download_file(s3_path.replace(f"s3://{BUCKET_NAME}/", ""), file_path)
    return file_path


# Assign GPU before loading the model
assign_available_gpus()


async def get_api_key(api_key: str = Depends(api_key_header)):
    """
    Dependency that validates the 'x-api-key' by calling an external service.
    """
    # 1. Check if the validation service URL is configured on the server.
    if not API_KEY_VALIDATION_URL:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API Key validation service is not configured on the server.",
        )

    # 2. The api_key should now be automatically provided by FastAPI
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="An API key is required in the 'x-api-key' header.",
        )

    # 3. Call the external service to validate the key.
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:  # Added timeout
            headers = {"x-api-key": api_key}
            response = await client.get(API_KEY_VALIDATION_URL, headers=headers)

            # 4. Check the validation result.
            if response.status_code == 200:
                return api_key
            elif response.status_code in (401, 403):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="The provided API key is invalid.",
                )
            else:
                print(
                    f"Validation service error: {response.status_code} - {response.text}"
                )
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="The API key validation service is currently unavailable.",
                )
    except httpx.TimeoutException:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="API key validation service timeout.",
        )
    except httpx.RequestError as e:
        print(f"Request error: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Could not connect to the API key validation service.",
        )


# Create separate routers for protected and public endpoints
protected_router = APIRouter(dependencies=[Depends(get_api_key)])
public_router = APIRouter()


def _parse_bounds(bounds):
    """Extract (minx, miny, maxx, maxy) from bounds object or tuple."""
    if hasattr(bounds, "left"):  # rasterio BoundingBox
        return bounds.left, bounds.bottom, bounds.right, bounds.top
    return bounds


def _reproject_to_4326(src_memfile):
    """
    Reproject in-memory raster to EPSG:4326.

    Args:
        src_memfile: Open MemoryFile dataset

    Returns:
        tuple: (MemoryFile, dataset) - caller must close both
    """
    dst_crs = "EPSG:4326"
    dst_transform, dst_width, dst_height = calculate_default_transform(
        src_memfile.crs,
        dst_crs,
        src_memfile.width,
        src_memfile.height,
        *src_memfile.bounds,
    )

    dst_profile = src_memfile.profile.copy()
    dst_profile.update(
        {
            "crs": dst_crs,
            "transform": dst_transform,
            "width": dst_width,
            "height": dst_height,
        }
    )

    dst_memfile = MemoryFile()
    with dst_memfile.open(**dst_profile) as dst:
        reproject(
            source=rasterio.band(src_memfile, 1),
            destination=rasterio.band(dst, 1),
            src_transform=src_memfile.transform,
            src_crs=src_memfile.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
        )

    return dst_memfile, dst_memfile.open()


def _resize_image(image, target_height, target_width):
    """Resize CHW image to target dimensions."""
    image = np.transpose(image, (1, 2, 0))  # CHW -> HWC
    image = np.resize(image, (target_height, target_width, image.shape[2]))
    return np.transpose(image, (2, 0, 1))  # HWC -> CHW


def save_cog(mosaic, profile, transform, filename):
    """
    Reproject raster to EPSG:4326 and save as a file.
    Args:
        mosaic (np.ndarray): The raster data.
        profile (dict): The rasterio profile.
        transform (affine.Affine): The rasterio transform.
        filename (str): The output filename.
    """
    src_profile = profile.copy()
    src_profile.update(
        {
            "driver": "GTiff",
            "height": mosaic.shape[0],
            "width": mosaic.shape[1],
            "transform": transform,
            "count": 1,
            "dtype": mosaic.dtype,
            "crs": profile.get("crs", "EPSG:4326"),
        }
    )

    with MemoryFile() as memfile:
        with memfile.open(**src_profile) as src:
            src.write(mosaic, 1)

            # Skip reprojection if already in EPSG:4326
            if str(src.crs) == "EPSG:4326":
                with rasterio.open(filename, "w", **src.profile) as out_raster:
                    out_raster.write(src.read())
                return filename

            # Reproject to EPSG:4326
            dst_memfile, dst = _reproject_to_4326(src)
            try:
                with rasterio.open(filename, "w", **dst.profile) as out_raster:
                    out_raster.write(dst.read())
            finally:
                dst.close()
                dst_memfile.close()

    return filename


def save_cog_cropped(
    mosaic,
    profile,
    transform,
    filename,
    crop_bounds,
    target_width=None,
    target_height=None,
):
    """
    Combined reproject + crop + save in a single pass (avoids double disk write).

    Args:
        mosaic (np.ndarray): The raster data (2D array).
        profile (dict): The rasterio profile from merge().
        transform (affine.Affine): The rasterio transform from merge().
        filename (str): The output filename.
        crop_bounds: Bounds to crop to (rasterio BoundingBox or [minx, miny, maxx, maxy]).
        target_width: Optional target width for resize.
        target_height: Optional target height for resize.

    Returns:
        str: The output filename.
    """
    minx, miny, maxx, maxy = _parse_bounds(crop_bounds)
    bbox_geom = box(minx, miny, maxx, maxy)

    # Build source profile from merge output
    src_profile = profile.copy()
    src_profile.update(
        {
            "driver": "GTiff",
            "height": mosaic.shape[0],
            "width": mosaic.shape[1],
            "transform": transform,
            "count": 1,
            "dtype": mosaic.dtype,
            "crs": profile.get("crs", "EPSG:4326"),
        }
    )

    with MemoryFile() as memfile:
        with memfile.open(**src_profile) as src:
            src.write(mosaic, 1)

            # Reproject if not already EPSG:4326
            if str(src.crs) != "EPSG:4326":
                dst_memfile, working_ds = _reproject_to_4326(src)
                should_close_dst = True
            else:
                working_ds = src
                dst_memfile = None
                should_close_dst = False

            try:
                # Crop in memory
                out_image, out_transform = mask(working_ds, [bbox_geom], crop=True)
                out_meta = working_ds.meta.copy()
            finally:
                if should_close_dst:
                    working_ds.close()
                    dst_memfile.close()

    # Resize if target dimensions specified
    if target_width and target_height:
        out_image = _resize_image(out_image, target_height, target_width)
        final_height, final_width = target_height, target_width
    else:
        final_height, final_width = out_image.shape[1], out_image.shape[2]

    out_meta.update(
        {"height": final_height, "width": final_width, "transform": out_transform}
    )

    # Single write to disk
    with rasterio.open(filename, "w", **out_meta) as dest:
        dest.write(out_image)

    return filename


def crop_file(filename, bbox, width=None, height=None, src_handle=None):
    """
    Reproject raster to EPSG:4326, crop to bbox, and save as COG

    Args:
        filename: output filename
        bbox: [minx, miny, maxx, maxy] in EPSG:4326 coordinates or rasterio.coords.BoundingBox
        width: target width
        height: target height
        src_handle: optional open rasterio file handle to reuse
    """
    # Handle both tuple/list and BoundingBox types
    if hasattr(bbox, "left"):  # rasterio BoundingBox
        minx, miny, maxx, maxy = bbox.left, bbox.bottom, bbox.right, bbox.top
    else:
        minx, miny, maxx, maxy = bbox

    # Pass Shapely geometry directly — avoids constructing a full GeoDataFrame
    bbox_geom = box(minx, miny, maxx, maxy)

    # Use provided handle or open new one
    should_close = False
    if src_handle is None:
        src_handle = rasterio.open(filename)
        should_close = True

    try:
        out_image, out_transform = mask(src_handle, [bbox_geom], crop=True)
        out_meta = src_handle.meta.copy()
    finally:
        if should_close:
            src_handle.close()

    # Decide target size
    if width and height:
        target_height, target_width = height, width
        # Reshape the output image to the target width and height
        out_image = np.transpose(out_image, (1, 2, 0))  # CHW -> HWC
        out_image = np.resize(
            out_image, (target_height, target_width, out_image.shape[2])
        )
        out_image = np.transpose(out_image, (2, 0, 1))  # HWC -> CHW
    else:
        target_height, target_width = out_image.shape[1], out_image.shape[2]

    out_meta.update(
        {"height": target_height, "width": target_width, "transform": out_transform}
    )

    with rasterio.open(filename, "w", **out_meta) as dest:
        dest.write(out_image)

    return filename


def post_process(detections, transform):
    contours, shape = PostProcess.prepare_contours(detections)
    detections = PostProcess.extract_shapes(detections, contours, transform, shape)
    # detections = PostProcess.remove_intersections(detections)
    return PostProcess.convert_to_geojson(detections)


def subset_geojson(geojson, bounding_box):
    geom = [shape(i["geometry"]) for i in geojson]
    geom = gpd.GeoDataFrame({"geometry": geom})
    bbox = {
        "type": "Polygon",
        "coordinates": [
            [
                [bounding_box[0], bounding_box[1]],
                [bounding_box[2], bounding_box[1]],
                [bounding_box[2], bounding_box[3]],
                [bounding_box[0], bounding_box[3]],
                [bounding_box[0], bounding_box[1]],
            ]
        ],
    }
    bbox = shape(bbox)
    bbox = gpd.GeoDataFrame({"geometry": [bbox]})
    return json.loads(geom.overlay(bbox, how="intersection").to_json())


def load_model(config_file_path, checkpoint_file_path, source="s3"):
    model_config_file_path = ""
    model_weights_path = ""
    if source == "s3":
        model_config_file_path = download_from_s3(config_file_path)
        model_weights_path = download_from_s3(checkpoint_file_path, "models")
    elif source == "huggingface":
        pass
    # Load the model only once
    usecase = inflection.singularize(USECASE)
    infer_classname = (
        f"{''.join([split.capitalize() for split in usecase.split('_')])}Infer"
    )
    model_module = importlib.import_module(f"lib.{usecase}_infer")
    infer = getattr(model_module, infer_classname)(
        model_config_file_path, model_weights_path
    )
    # Cache postprocess signature check once at load time — avoids inspect overhead per inference call
    sig = inspect.signature(infer.postprocess)
    infer._postprocess_accepts_dims = (
        "source_width" in sig.parameters and "source_height" in sig.parameters
    )
    return {USECASE: infer}


def ensure_model_loaded(model_id):
    global MODEL

    # Fast check: If it's already there, return it immediately
    if model_id in MODEL:
        print(f"Model {model_id} already loaded, using cached version.")
        return MODEL[model_id]

    # If not, acquire lock so only ONE thread loads
    with model_lock:
        # Double-check inside lock (another thread might have finished while we waited)
        if model_id not in MODEL:
            print(f"--- [LOCK ACQUIRED] Loading model {model_id} ---")
            # Reuse your existing loading logic
            loaded_dict = load_model(CONFIG_PATH, MODEL_WEIGHT_PATH)
            MODEL.update(loaded_dict)

    return MODEL.get(model_id)


def _create_memfile_dataset(args):
    """Create a MemoryFile dataset for a single tile (used in parallel mosaic assembly)."""
    result, profile = args
    profile = profile.copy()
    profile.update({"count": 1, "dtype": "float32", "nodata": 0})
    memfile = MemoryFile()
    with memfile.open(**profile) as dst:
        dst.write(result, 1)
    return memfile, memfile.open()


def infer(filename, scale, model_id, bounding_box, date, qa_flags, timeseries=False):
    inference = ensure_model_loaded(model_id)
    if not inference:
        response = {"statusCode": 422}
        return JSONResponse(content=jsonable_encoder(response))

    # Cache source file metadata to avoid multiple file opens
    with rasterio.open(filename) as src:
        source_bounds = src.bounds
        source_width = src.profile["width"]
        source_height = src.profile["height"]

    data_preparer = DataPreparer(
        filename, overlap=0, scale=scale, qa_flags=qa_flags, timeseries=timeseries
    )

    batch_start_time = time.time()
    # Use DataLoader-based inference with prefetching for better GPU utilization
    results, profiles = inference.infer_dataloader(
        data_preparer, date, num_workers=2, prefetch_factor=2
    )
    batch_infer_time = time.time() - batch_start_time
    print(f"Inference time for batch: {batch_infer_time:.2f} seconds")

    mosaic_start_time = time.time()

    # Parallel mosaic assembly - create MemoryFile datasets concurrently
    with ThreadPoolExecutor(max_workers=min(8, len(results))) as pool:
        memfile_results = list(
            pool.map(_create_memfile_dataset, zip(results, profiles))
        )

    memory_files, datasets = zip(*memfile_results) if memfile_results else ([], [])
    memory_files, datasets = list(memory_files), list(datasets)

    # Merge all tiles into single mosaic
    mosaic, transform = merge(datasets)

    # Clean up MemoryFiles
    for ds in datasets:
        ds.close()
    for memfile in memory_files:
        memfile.close()
    del datasets, memory_files

    # Combined save + crop in single disk write (avoids writing twice)
    prediction_filename = f"{PREDICTION_FOLDER}/{mosaic_start_time}-predictions.tif"
    # Use last profile as template (all tiles have same CRS/dtype)
    prediction_filename = save_cog_cropped(
        mosaic[0],
        profiles[-1],
        transform,
        prediction_filename,
        crop_bounds=source_bounds,
        target_width=source_width,
        target_height=source_height,
    )

    del mosaic, results, profiles
    # Removed gc.collect() - causes unpredictable pauses that contribute to timing variance
    # Python's refcount-based cleanup handles the deleted objects immediately
    print("!!! Mosaic and Save COG Time:", time.time() - mosaic_start_time)

    crop_post_start_time = time.time()

    # Pass cached dimensions to postprocess to avoid reopening file
    if inference._postprocess_accepts_dims:
        postprocessed_filename = inference.postprocess(
            bounding_box,
            date,
            prediction_filename,
            filename,
            source_width=source_width,
            source_height=source_height,
        )
    else:
        postprocessed_filename = inference.postprocess(
            bounding_box, date, prediction_filename, filename
        )
    print("!!! Crop and Postprocess Time:", time.time() - crop_post_start_time)
    qa_postprocess_start_time = time.time()

    # Collect any DEM-based postprocessing artifacts attached by the model
    postprocess_artifacts = getattr(inference, "postprocess_artifacts", {}) or {}

    # Generate QA TIFs (sync, CPU-bound)
    qa_flag_tifs = inference.qa_flags_to_tif(filename, qa_flags, timeseries=timeseries)

    # Calculate stats (sync, CPU-bound)
    stats = inference.calculate_area_from_mask(
        postprocessed_filename, mask_values=range(1, NUM_CLASSES)
    )

    print("!!! QA generation + stats Time:", time.time() - qa_postprocess_start_time)

    # Return data for async upload phase
    return {
        "model_id": model_id,
        "postprocessed_filename": postprocessed_filename,
        "qa_flag_tifs": qa_flag_tifs,
        "postprocess_artifacts": postprocess_artifacts,
        "stats": stats,
    }


async def upload_results_async(infer_result: dict) -> dict:
    """
    Async upload phase: uploads prediction, QA masks, and postprocess artifacts concurrently.
    """
    upload_start = time.time()

    model_id = infer_result["model_id"]
    postprocessed_filename = infer_result["postprocessed_filename"]
    qa_flag_tifs = infer_result["qa_flag_tifs"]
    postprocess_artifacts = infer_result["postprocess_artifacts"]
    stats = infer_result["stats"]

    # Run all uploads concurrently using asyncio
    s3_upload_task = asyncio.to_thread(upload_cog_to_s3, postprocessed_filename)
    qa_upload_task = async_upload_many_to_s3(qa_flag_tifs, skip_cog=True)
    supplement_upload_task = async_upload_many_to_s3(
        postprocess_artifacts, skip_cog=False
    )

    s3_link, qa_links, supplement_links = await asyncio.gather(
        s3_upload_task, qa_upload_task, supplement_upload_task
    )

    print("!!! Async Upload Time:", time.time() - upload_start)

    return {
        model_id: {
            "s3_link": s3_link,
            "qa_links": qa_links,
            "stats": stats,
            "postprocess_links": supplement_links,
        }
    }


# Define a model for the POST request body
class InvocationData(BaseModel):
    filename: str
    scale: Optional[bool] = False
    model_id: str
    bounding_box: list[float]
    date: Optional[str] = None
    qa_flags: Optional[list[str]] = (["cloud", "shadow", "adjacent_cloud"],)
    timeseries: Optional[bool] = False


_gpu_lock = threading.Lock()


@public_router.post("/invocations")
async def infer_from_model(invocation_data: InvocationData = Body(...)):

    print(f"====STARTED at {time.strftime('%X %x %Z')}=====")

    filename = invocation_data.filename
    print(
        f"Received inference request for model: {invocation_data.model_id} on data: {filename}"
    )
    t0 = time.time()

    def _locked_infer():
        with _gpu_lock:
            return infer(
                filename,
                invocation_data.scale,
                invocation_data.model_id,
                invocation_data.bounding_box,
                invocation_data.date,
                invocation_data.qa_flags,
                bool(invocation_data.timeseries),
            )

    infer_result = await asyncio.to_thread(_locked_infer)
    print(f"Inference time: {time.time() - t0:.2f} seconds")

    final_geojson = await upload_results_async(infer_result)

    print(f"Total time: {time.time() - t0:.2f} seconds")
    print(f"=====FINISHED at {time.strftime('%X %x %Z')}===")

    return JSONResponse(content=jsonable_encoder(final_geojson))


# Public endpoints (no API key required)
@public_router.get("/ping")
async def ping(request: Request):
    return {"successCode": 200, "message": "pong"}


@public_router.get("/health")
async def health():
    return {"successCode": 200, "status": "healthy"}


# Include both routers in the v1_api
v1_api.include_router(protected_router)
v1_api.include_router(public_router)
app.mount("/api/v1", v1_api)
