import asyncio
import boto3
import os

from concurrent.futures import ThreadPoolExecutor, as_completed
from os import path
from glob import glob
from rasterio.io import MemoryFile
from rio_cogeo.cogeo import cog_translate
from rio_cogeo.profiles import cog_profiles

from lib.consts import BUCKET_NAME, MODEL_PATH


def get_boto3_session():
    # Assume the "notebookAccessRole" role we created using AWS CDK.
    return boto3.session.Session()


def download_data(data, split):
    split_folder = f"/opt/ml/data/{split}"
    if not (os.path.exists(split_folder)):
        os.makedirs(split_folder)
    session = get_boto3_session()
    s3_connection = session.resource("s3")
    splits = data.split("/")
    bucket = s3_connection.Bucket(BUCKET_NAME)
    objects = list(bucket.objects.filter(Prefix="/".join(splits[3:] + [split])))
    print("Downloading files:", data, split)
    for iter_object in objects:
        splits = iter_object.key.split("/")
        if splits[-1]:
            filename = f"{split_folder}/{splits[-1]}"
            bucket.download_file(iter_object.key, filename)
    print("Finished downloading files.")


def save_model_artifacts(s3_connection, model_artifacts_path):
    if path.exists(model_artifacts_path):
        print("files", glob(f"{model_artifacts_path}/*"))
        for model_file in glob(f"{model_artifacts_path}/best*"):
            model_name = model_file.split("/")[-1]
            model_name = os.environ.get("MODEL_NAME", model_name)
            model_name = MODEL_PATH.format(model_name=model_name)
            print(f"Uploading model to s3: s3://{BUCKET_NAME}/{model_name}")
            s3_connection.meta.client.upload_file(model_file, BUCKET_NAME, model_name)


def upload_cog_to_s3(
    filename: str, prefix: str = "predictions", max_retries: int = 3
) -> str:
    """
    Convert a GeoTIFF to COG in-memory and upload to S3.

    Returns the s3:// URL.
    """
    if not BUCKET_NAME:
        # Fail gracefully in local setups without S3
        return filename

    # Verify file exists and is readable before attempting COG translation
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")

    output_profile = cog_profiles.get("deflate")
    output_profile.update(dict(BIGTIFF="IF_SAFER"))

    config = dict(
        GDAL_NUM_THREADS="ALL_CPUS",
        GDAL_TIFF_INTERNAL_MASK=True,
        GDAL_TIFF_OVR_BLOCKSIZE="512",
    )

    basename = os.path.basename(filename)
    s3_prefix = f"{prefix}/{basename}"

    # Retry logic for transient I/O errors
    last_exception = None
    for attempt in range(max_retries):
        try:
            with MemoryFile() as memory_file:
                cog_translate(
                    filename,
                    memory_file.name,
                    output_profile,
                    config=config,
                    quiet=True,
                    in_memory=True,
                )
                connection = boto3.client("s3")
                connection.upload_fileobj(memory_file, BUCKET_NAME, s3_prefix)
            return f"s3://{BUCKET_NAME}/{s3_prefix}"
        except Exception as e:
            last_exception = e
            if attempt < max_retries - 1:
                import time

                # Brief delay before retry to allow any pending I/O to complete
                time.sleep(0.5 * (attempt + 1))
            continue

    # All retries failed
    raise last_exception


def upload_raw_to_s3(
    filename: str, prefix: str = "predictions", max_retries: int = 3
) -> str:
    """
    Upload a file directly to S3 without COG translation.

    Use this for simple files like QA masks (uint8) where COG overviews
    provide no benefit but add significant processing time.

    Returns the s3:// URL.
    """
    if not BUCKET_NAME:
        return filename

    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")

    basename = os.path.basename(filename)
    s3_prefix = f"{prefix}/{basename}"

    last_exception = None
    for attempt in range(max_retries):
        try:
            connection = boto3.client("s3")
            connection.upload_file(filename, BUCKET_NAME, s3_prefix)
            return f"s3://{BUCKET_NAME}/{s3_prefix}"
        except Exception as e:
            last_exception = e
            if attempt < max_retries - 1:
                import time

                time.sleep(0.5 * (attempt + 1))
            continue

    raise last_exception


def upload_many_to_s3(
    paths: dict[str, str],
    max_workers: int = 4,
    skip_cog: bool = False,
) -> dict[str, str]:
    """Upload many local paths to S3 in parallel; returns key->s3_link.

    Args:
        paths: dict mapping key names to local file paths
        max_workers: number of parallel upload threads
        skip_cog: if True, upload raw files without COG translation (faster for QA masks)
    """
    upload_fn = upload_raw_to_s3 if skip_cog else upload_cog_to_s3
    results: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_key = {
            executor.submit(upload_fn, path): key for key, path in paths.items()
        }
        for future in as_completed(future_to_key):
            key = future_to_key[future]
            results[key] = future.result()
    return results


async def async_upload_many_to_s3(
    paths: dict[str, str],
    skip_cog: bool = False,
) -> dict[str, str]:
    """
    Async version: Upload many local paths to S3 concurrently using asyncio.to_thread.

    Wraps the existing sync upload functions to run in thread pool without blocking
    the event loop.

    Args:
        paths: dict mapping key names to local file paths
        skip_cog: if True, upload raw files without COG translation (faster for QA masks)

    Returns:
        dict mapping key names to s3:// URLs
    """
    upload_fn = upload_raw_to_s3 if skip_cog else upload_cog_to_s3

    async def _upload_one(key: str, filepath: str) -> tuple[str, str]:
        url = await asyncio.to_thread(upload_fn, filepath)
        return key, url

    tasks = [_upload_one(key, filepath) for key, filepath in paths.items()]
    results_list = await asyncio.gather(*tasks)
    return dict(results_list)
