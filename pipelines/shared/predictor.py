import boto3
import gc
import geopandas as gpd
import httpx
import importlib
import inflection
import json
import logging
import numpy as np
import os
import rasterio
import time
import torch

from fastapi import FastAPI, Request, APIRouter, status, Response, Body, Depends, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader

from lib.consts import BUCKET_NAME, LAYERS, CONFIG_PATH, MODEL_WEIGHT_PATH, USECASE, DOWNLOAD_FOLDER
from lib.data_preparer import DataPreparer
from lib.post_process import PostProcess
from lib.utils import get_boto3_session

from pydantic import BaseModel

from rasterio.io import MemoryFile
from rasterio.mask import mask
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject, Resampling

from rio_cogeo.cogeo import cog_translate
from rio_cogeo.profiles import cog_profiles

from shapely.geometry import shape, box

from typing import Optional

PREDICTION_FOLDER = f"{DOWNLOAD_FOLDER}/predictions"
os.makedirs(PREDICTION_FOLDER, exist_ok=True)

# This will be served by the FastAPI as a container
# Re-enable docs to see the authorization feature
# Create without docs
app = FastAPI(
    docs_url=None,
    redoc_url=None,
)

# Todo Provide a better title
v1_api = FastAPI(
    title="Predictor API for Floods",
    description="Predictor API Floods (MVP)",
    version="0.0.1"
)

# --- Start of Modified Security Block ---

# This is the URL of your external validation service.
API_KEY_VALIDATION_URL = os.getenv("API_KEY_VALIDATION_URL", "https://dev.fm.dsig.net/api/validate")

# Fixed: Remove auto_error=False to make it work with FastAPI's authorization UI
api_key_header = APIKeyHeader(name="x-api-key")

def download_from_s3(s3_path, download_path='config'):
    download_path = f"{DOWNLOAD_FOLDER}/{download_path}"
    session = get_boto3_session()
    s3_connection = session.resource('s3')
    bucket = s3_connection.Bucket(BUCKET_NAME)
    filename = s3_path.split('/')[-1]
    file_path = f"{download_path}/{filename}"
    if not(os.path.exists(file_path)):
        os.makedirs(download_path, exist_ok=True)
        os.makedirs('predictions', exist_ok=True)
        bucket.download_file(s3_path.replace(f's3://{BUCKET_NAME}/', ''), file_path)
    return file_path


def load_model(config_file_path, checkpoint_file_path, source='s3'):
    if source == 's3':
        model_config_file_path = download_from_s3(config_file_path)
        model_weights_path = download_from_s3(checkpoint_file_path, 'models')
    elif source == 'huggingface':
        pass
    # Load the model only once
    usecase = inflection.singularize(USECASE)
    infer_classname = f"{"".join([split.capitalize() for split in usecase.split('_')])}Infer"
    model_module = importlib.import_module(f"lib.{usecase}_infer")
    infer = getattr(model_module, infer_classname)(model_config_file_path, model_weights_path)
    return { USECASE: infer }

MODEL = load_model(CONFIG_PATH, MODEL_WEIGHT_PATH)

async def get_api_key(api_key: str = Depends(api_key_header)):
    """
    Dependency that validates the 'x-api-key' by calling an external service.
    """
    # 1. Check if the validation service URL is configured on the server.
    if not API_KEY_VALIDATION_URL:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API Key validation service is not configured on the server."
        )

    # 2. The api_key should now be automatically provided by FastAPI
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="An API key is required in the 'x-api-key' header."
        )

    # 3. Call the external service to validate the key.
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:  # Added timeout
            headers = {'x-api-key': api_key}
            response = await client.get(API_KEY_VALIDATION_URL, headers=headers)

            # 4. Check the validation result.
            if response.status_code == 200:
                return api_key
            elif response.status_code in (401, 403):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="The provided API key is invalid."
                )
            else:
                print(f"Validation service error: {response.status_code} - {response.text}")
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="The API key validation service is currently unavailable."
                )
    except httpx.TimeoutException:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="API key validation service timeout."
        )
    except httpx.RequestError as e:
        print(f"Request error: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Could not connect to the API key validation service."
        )

# Create separate routers for protected and public endpoints
protected_router = APIRouter(dependencies=[Depends(get_api_key)])
public_router = APIRouter()

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
    src_profile.update({
        'driver': 'GTiff',
        'height': mosaic.shape[0],
        'width': mosaic.shape[1],
        'transform': transform,
        'count': 1,
        'dtype': mosaic.dtype,
        'crs': profile.get('crs', 'EPSG:3857') # Assuming default CRS if not provided
    })

    with MemoryFile() as memfile:
        with memfile.open(**src_profile) as src:
            src.write(mosaic, 1)

            # Reproject
            if src.crs == 'EPSG:4326':
                # Directly save as COG if already in EPSG:4326
                with rasterio.open(filename, 'w', **src.profile) as out_raster:
                    out_raster.write(src.read())
                return filename
            dst_crs = 'EPSG:4326'

            dst_transform, dst_width, dst_height = calculate_default_transform(
                src.crs, dst_crs, src.width, src.height, *src.bounds)

            dst_profile = src.profile.copy()
            dst_profile.update({
                'crs': dst_crs,
                'transform': dst_transform,
                'width': dst_width,
                'height': dst_height
            })

            with MemoryFile() as dst_memfile:
                with dst_memfile.open(**dst_profile) as dst:
                    reproject(
                        source=rasterio.band(src, 1),
                        destination=rasterio.band(dst, 1),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=dst_transform,
                        dst_crs=dst_crs,
                        resampling=Resampling.bilinear)

                with dst_memfile.open() as reprojected_raster:
                    with rasterio.open(filename, 'w', **reprojected_raster.profile) as out_raster:
                        out_raster.write(reprojected_raster.read())
    return filename

def crop_file(filename, bbox, width=None, height=None):
    """
    Reproject raster to EPSG:4326, crop to bbox, and save as COG

    Args:
        filename: output filename
        bbox: [minx, miny, maxx, maxy] in EPSG:4326 coordinates
        width: target width
        height: target height
    """
    # Extract bbox coordinates
    minx, miny, maxx, maxy = bbox
    # Create bbox geometry
    bbox_geom = box(minx, miny, maxx, maxy)
    clip_geom = gpd.GeoDataFrame({'geometry': [bbox_geom]}, crs='EPSG:4326')

    with rasterio.open(filename) as src:
        out_image, out_transform = mask(src, clip_geom.geometry, crop=True)
        out_meta = src.meta.copy()

    out_meta.update({
        "height": out_image.shape[1],
        "width": out_image.shape[2],
        "transform": out_transform
    })

    with rasterio.open(filename, "w", **out_meta) as dest:
        if width and height:
            # Reshape the output image to the target width and height
            out_image = np.transpose(out_image, (1, 2, 0)) # HWC
            out_image = np.resize(out_image, (height, width, out_image.shape[2]))
            out_image = np.transpose(out_image, (2, 0, 1)) # CHW
            out_meta.update({
                "height": height,
                "width": width,
            })

        dest.write(out_image)

    return filename


def upload_to_s3(filename):
    output_profile = cog_profiles.get('deflate')
    output_profile.update(dict(BIGTIFF="IF_SAFER"))

    config = dict(
        GDAL_NUM_THREADS="ALL_CPUS",
        GDAL_TIFF_INTERNAL_MASK=True,
        GDAL_TIFF_OVR_BLOCKSIZE="512",
    )
    s3_prefix = f"predictions/{filename.split('/')[-1]}"
    with MemoryFile() as memory_file:
        cog_translate(
            filename,
            memory_file.name,
            output_profile,
            config=config,
            quiet=True,
            in_memory=True,
        )
        connection = boto3.client('s3')
        connection.upload_fileobj(memory_file, BUCKET_NAME, s3_prefix)

    return f"s3://{BUCKET_NAME}/{s3_prefix}"

def post_process(detections, transform):
    contours, shape = PostProcess.prepare_contours(detections)
    detections = PostProcess.extract_shapes(detections, contours, transform, shape)
    # detections = PostProcess.remove_intersections(detections)
    return PostProcess.convert_to_geojson(detections)

def subset_geojson(geojson, bounding_box):
    geom = [shape(i['geometry']) for i in geojson]
    geom = gpd.GeoDataFrame({'geometry': geom})
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
    bbox = gpd.GeoDataFrame({'geometry': [bbox]})
    return json.loads(geom.overlay(bbox, how='intersection').to_json())

def infer(filename, scale, model_id, bounding_box, date):
    if model_id not in MODEL:
        response = {'statusCode': 422}
        return JSONResponse(content=jsonable_encoder(response))
    inference = MODEL[model_id]
    all_tiles = list()
    geojson_list = list()
    geojson = {'type': 'FeatureCollection', 'features': []}

    start_time = time.time()
    results = list()
    profiles = list()
    s3_link = ''
    tiles_generator = DataPreparer(filename, overlap=100, scale=scale).generate_tiles()
    torch.cuda.synchronize()
    with torch.no_grad():
        for tiles in tiles_generator:
            batch_results, batch_profiles = inference.infer(tiles, date)
            results.extend(batch_results)
            # profile is ofset by some value. need to debug
            profiles.extend(batch_profiles)
    memory_files = list()
    torch.cuda.empty_cache()
    for index, profile in enumerate(profiles):
        memfile = MemoryFile()
        profile.update({
            'count': 1,
            'dtype': 'float32',
            'nodata': 0
        })
        with memfile.open(**profile) as memoryfile:
            memoryfile.write(results[index], 1)
        memory_files.append(memfile.open())
    mosaic, transform = merge(memory_files)
    [memfile.close() for memfile in memory_files]
    prediction_filename = f"{PREDICTION_FOLDER}/{start_time}-predictions.tif"

    prediction_filename = save_cog(mosaic[0], profile, transform, prediction_filename)
    with rasterio.open(filename) as src:
        bounds = src.bounds
    prediction_filename = crop_file(prediction_filename, bounds)
    postprocessed_filename = inference.postprocess(bounding_box, date, prediction_filename, filename)
    s3_link = upload_to_s3(prediction_filename)

    geojson = post_process(mosaic[0], transform)

    for geometry in geojson:
        updated_geometry = PostProcess.convert_geojson(geometry)
        geojson_list.append(updated_geometry)
    geojson = subset_geojson(geojson_list, bounding_box)
    print("!!! Infer Time:", time.time() - start_time)
    del inference
    gc.collect()

    return {
        model_id: {'s3_link': s3_link, 'geojson': geojson}
    }

# Define a model for the POST request body
class InvocationData(BaseModel):
    filename: str
    scale: Optional[bool] = False
    model_id: str
    bounding_box: list[float]
    date: Optional[str] = None
    qa_flags: Optional[list[str]] = ['cloud', 'shadow', 'adjacent_cloud']


# Protected endpoints (require API key)
@public_router.post('/invocations')
async def infer_from_model(invocation_data: InvocationData = Body(...)):
    filename = invocation_data.filename
    final_geojson = infer(
        filename=filename,
        scale=invocation_data.scale,
        model_id=invocation_data.model_id,
        bounding_box=invocation_data.bounding_box,
        date=invocation_data.date
    )
    return JSONResponse(content=jsonable_encoder(final_geojson))

# Public endpoints (no API key required)
@public_router.get('/ping')
async def ping(request: Request):
    return { "successCode": 200, "message": "pong"}

@public_router.get("/health")
async def health():
    return {"successCode": 200, "status": "healthy"}

# Include both routers in the v1_api
v1_api.include_router(protected_router)
v1_api.include_router(public_router)
app.mount("/api/v1", v1_api)
