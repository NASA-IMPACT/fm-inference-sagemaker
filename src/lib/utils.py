import httpx
import os

from fastapi import Depends, HTTPException, status
from fastapi.security import APIKeyHeader

API_KEY_VALIDATION_URL = os.getenv(
    "API_KEY_VALIDATION_URL", "https://dev.fm.dsig.net/api/validate"
)
api_key_header = APIKeyHeader(name="x-api-key")


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


#
