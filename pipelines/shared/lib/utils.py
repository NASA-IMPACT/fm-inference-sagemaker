import boto3
import os
import httpx


from fastapi import APIRouter, Depends, Query, status
from os import path
from glob import glob
from lib.consts import BUCKET_NAME, MODEL_PATH, API_KEY_VALIDATION_URL


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
