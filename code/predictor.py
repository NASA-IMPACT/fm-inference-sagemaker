import boto3
import json
import os
import gc
import geopandas as gpd
import rasterio
import time
import torch

from fastapi import FastAPI, Request, APIRouter, status, Response, Body, Depends, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader

from lib.downloader import  Downloader
from lib.infer import Infer
from lib.post_process import PostProcess
from lib.consts import BUCKET_NAME, LAYERS, USECASE

from lib.utils import get_boto3_session

from rasterio.io import MemoryFile
from rasterio.merge import merge
from rio_cogeo.cogeo import cog_translate
from rio_cogeo.profiles import cog_profiles

from shapely.geometry import shape

from pydantic import BaseModel
from typing import Optional
import httpx

# This will be served by the FastAPI as a container
# Re-enable docs to see the authorization feature
# Create without docs
app = FastAPI(
    docs_url=None,
    redoc_url=None,
)

# Todo Provide a better title
v1_api = FastAPI(
    title="Predictor API - V1",
    description="Predictor API for NASA IMPACT (MVP)",
    version="1.0.1"
)

# --- Start of Modified Security Block ---

# This is the URL of your external validation service.
API_KEY_VALIDATION_URL = os.getenv("API_KEY_VALIDATION_URL", "https://dev.fm.dsig.net/api/validate")

# Fixed: Remove auto_error=False to make it work with FastAPI's authorization UI
api_key_header = APIKeyHeader(name="x-api-key")

MODEL = None

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

# ... [rest of your existing functions remain the same] ...

def download_from_s3(s3_path, download_path='config'):
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

def load_model(config_filename, checkpoint_file):
    model_config_file_path = download_from_s3(config_filename)
    model_weights_path = download_from_s3(checkpoint_file, 'models')
    infer = Infer(model_config_file_path, model_weights_path)
    return { USECASE: infer }

def download_files(infer_date, layer, bounding_box):
    downloader = Downloader(infer_date, layer)
    result = downloader.download_tiles(bounding_box)
    return result

def save_cog(mosaic, profile, transform, filename):
    profile.update(
        {
            "driver": "GTiff",
            "height": mosaic.shape[0],
            "width": mosaic.shape[1],
            "transform": transform,
            "dtype": 'float32',
            "count": 1,
        }
    )
    with rasterio.open(filename, 'w', **profile) as raster:
        raster.write(mosaic, 1)
    output_profile = cog_profiles.get('deflate')
    output_profile.update(dict(BIGTIFF="IF_SAFER"))

    config = dict(
        GDAL_NUM_THREADS="ALL_CPUS",
        GDAL_TIFF_INTERNAL_MASK=True,
        GDAL_TIFF_OVR_BLOCKSIZE="512",
    )
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
        connection.upload_fileobj(memory_file, BUCKET_NAME, filename)

    return f"s3://{BUCKET_NAME}/{filename}"

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

def batch(tiles, spacing=60):
    spacing = max(spacing, 120)
    length = len(tiles)
    for tile in range(0, length, spacing):
        yield tiles[tile : min(tile + spacing, length)]

def infer(model_id, infer_date, bounding_box, config_filename, checkpoint_file, terramind=False, file_links=[]):
    global MODEL
    MODEL = MODEL or load_model(config_filename=config_filename, checkpoint_file=checkpoint_file)
    if model_id not in MODEL:
        response = {'statusCode': 422}
        return JSONResponse(content=jsonable_encoder(response))
    inference = MODEL[model_id]
    all_tiles = list()
    geojson_list = list()
    geojson = {'type': 'FeatureCollection', 'features': []}
    
    if terramind:
        for file_link in file_links:
            all_tiles.append(download_from_s3(file_link, '/opt/ml/data'))
    else:
        for layer in LAYERS:
            tiles = download_files(infer_date, layer, bounding_box)
            for tile in tiles:
                tile_name = tile.replace('.tif', '_scaled.tif')
                all_tiles.append(tile_name)

    start_time = time.time()
    results = list()
    profiles = list()
    s3_link = ''
    if all_tiles:
        try:
            torch.cuda.synchronize()
            with torch.no_grad():
                for tiles in batch(all_tiles):
                    batch_results, batch_profiles = inference.infer(tiles, terramind)
                    results.extend(batch_results)
                    profiles.extend(batch_profiles)
            memory_files = list()
            torch.cuda.empty_cache()
            for index, profile in enumerate(profiles):
                memfile = MemoryFile()
                profile.update({
                    'count': 1,
                    'dtype': 'float32'
                })
                with memfile.open(**profile) as memoryfile:
                    memoryfile.write(results[index][0], 1)
                memory_files.append(memfile.open())

            mosaic, transform = merge(memory_files)

            [memfile.close() for memfile in memory_files]
            prediction_filename = f"predictions/{start_time}-predictions.tif"

            s3_link = save_cog(mosaic[0], profile, transform, prediction_filename)

            geojson = post_process(mosaic[0], transform)

            for geometry in geojson:
                updated_geometry = PostProcess.convert_geojson(geometry)
                geojson_list.append(updated_geometry)
            geojson = subset_geojson(geojson_list, bounding_box)
        except Exception as e:
            print('!!! infer error', infer_date, model_id, bounding_box, e)
            torch.cuda.empty_cache()
        print("!!! Infer Time:", time.time() - start_time)
    del inference
    gc.collect()


    return {
        model_id: {'s3_link': s3_link, 'predictions': geojson}
    }

# Define a model for the POST request body
class InvocationData(BaseModel):
    bounding_box: list[float]
    date: str
    model_id: str
    config_filename: str
    checkpoint_file: str
    terramind: Optional[bool] = False
    file_links: Optional[list[str]] = []

# Protected endpoints (require API key)
@protected_router.post('/invocations')
async def infer_from_model(invocation_data: InvocationData = Body(...)):
    model_id = invocation_data.model_id
    infer_date = invocation_data.date
    bounding_box = invocation_data.bounding_box
    terramind = invocation_data.terramind
    file_links = invocation_data.file_links
    config_filename = invocation_data.config_filename
    checkpoint_file = invocation_data.checkpoint_file
    final_geojson = infer(model_id, infer_date, bounding_box, config_filename=config_filename, checkpoint_file=checkpoint_file, terramind=terramind, file_links=file_links)
    return JSONResponse(content=jsonable_encoder(final_geojson))

# Public endpoints (no API key required)
@public_router.get('/ping')
async def ping(request: Request):
    return { "successCode": 200, "message": "pong"}

@public_router.get("/health")
async def health():
    return {"successCode": 200, "status": "healthy" , "service", "up"}

# Include both routers in the v1_api
v1_api.include_router(protected_router)
v1_api.include_router(public_router)
app.mount("/api/predict", v1_api)
