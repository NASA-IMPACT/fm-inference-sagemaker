import boto3
import os

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
    s3_connection = session.resource('s3')
    splits = data.split('/')
    bucket = s3_connection.Bucket(BUCKET_NAME)
    objects = list(bucket.objects.filter(Prefix="/".join(splits[3:] + [split])))
    print("Downloading files:", data, split)
    for iter_object in objects:
        splits = iter_object.key.split('/')
        if splits[-1]:
            filename = f"{split_folder}/{splits[-1]}"
            bucket.download_file(iter_object.key, filename)
    print("Finished downloading files.")


def save_model_artifacts(s3_connection, model_artifacts_path):
    if path.exists(model_artifacts_path):
        print('files', glob(f"{model_artifacts_path}/*"))
        for model_file in glob(f"{model_artifacts_path}/best*"):
            model_name = model_file.split('/')[-1]
            model_name = os.environ.get('MODEL_NAME', model_name)
            model_name = MODEL_PATH.format(model_name=model_name)
            print(f"Uploading model to s3: s3://{BUCKET_NAME}/{model_name}")
            s3_connection.meta.client.upload_file(model_file, BUCKET_NAME, model_name)


def upload_cog_to_s3(filename: str, prefix: str = "predictions") -> str:
    """
    Convert a GeoTIFF to COG in-memory and upload to S3.

    Returns the s3:// URL. 
    """
    if not BUCKET_NAME:
        # Fail gracefully in local setups without S3
        return filename

    output_profile = cog_profiles.get("deflate")
    output_profile.update(dict(BIGTIFF="IF_SAFER"))

    config = dict(
        GDAL_NUM_THREADS="ALL_CPUS",
        GDAL_TIFF_INTERNAL_MASK=True,
        GDAL_TIFF_OVR_BLOCKSIZE="512",
    )

    basename = os.path.basename(filename)
    s3_prefix = f"{prefix}/{basename}"

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
