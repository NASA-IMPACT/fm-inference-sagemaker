from fastapi import FastAPI, Request, Depends, status, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from contextlib import asynccontextmanager
import os
from datetime import datetime, timezone, timedelta
from jose import JWTError, jwt
import logging
from typing import Any, Optional, Dict
from pydantic import BaseModel, Field
import base64
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from .api.v1 import (
    create_inference_router,
    models_router,
    preloaded_events_router
)

class TokenRequest(BaseModel):
    grouplist: list[str]
    expires_in_days: int = Field(default=1, ge=1)
                                 

SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "676b780b2067723bef14910a7ad9e0ae5e3a14725dc1d7f08bb6fec6ff1e0e6a")
ALGORITHM = os.environ.get("JWT_ALGORITHM", "HS256")


async def require_alb_authentication(request: Request) -> Dict[str, Any]:
    """Dependency to ensure a user is authenticated by the ALB."""
    access_token = request.headers.get("x-amzn-oidc-accesstoken")
    if not access_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User is not authenticated via ALB/Cognito."
        )
    return get_jwt_payload(access_token)


def get_jwt_payload(token: str) -> Dict[str, Any]:
    """Decodes the payload from a JWT without verification (trusting the ALB)."""
    try:
        _, payload_b64, _ = token.split('.')
        payload_b64 += '=' * (-len(payload_b64) % 4)
        decoded_payload = base64.b64decode(payload_b64).decode('utf-8')
        return json.loads(decoded_payload)
    except Exception as e:
        logger.error(f"Error decoding ALB JWT payload: {e}")
        raise HTTPException(status_code=401, detail="Invalid ALB token format")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    print("Starting FM Inference Service.")

    # Create database tables (if they don't exist)
    try:
        from .db.database import engine, Base
        Base.metadata.create_all(bind=engine)
        print("Database tables created/verified successfully")
    except Exception as e:
        print(f"Warning: Could not create database tables: {e}")

    yield

    # Shutdown
    print("Shutting down FM Inference Service...")


root_path = os.environ.get("FASTAPI_ROOT_PATH", "/api/predict")

app = FastAPI(
    title="FM Inference Service",
    description="REST API for managing finetuned models and inferences.",
    version="0.0.1",
    lifespan=lifespan,
    root_path=root_path
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Authentication functions (keep these in main.py) ---
oauth2_scheme = HTTPBearer(auto_error=False)

async def verify_custom_token(
    creds: Optional[HTTPAuthorizationCredentials] = Depends(oauth2_scheme)
) -> Optional[Dict[str, Any]]:
    """Dependency to validate the custom-generated bearer token."""
    if not creds:
        return None
    
    token = creds.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None
    
async def general_access_dependency(
    custom_token_payload: Optional[Dict[str, Any]] = Depends(verify_custom_token),
) -> Dict[str, Any]:
    """General authentication dependency that doesn't require a specific group."""
    if custom_token_payload:
        logger.info(f"Authenticating via custom JWT for user '{custom_token_payload.get('sub')}'.")
        return custom_token_payload

    raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated. Provide a valid bearer token."
        )

# Include v1 API routers
inference_router = create_inference_router(general_access_dependency)
app.include_router(inference_router)
app.include_router(models_router)
app.include_router(preloaded_events_router)


@app.post("/create-token", tags=["Authentication"])
async def create_token(
    body: TokenRequest,
    claims: Dict[str, Any] = Depends(require_alb_authentication),
    
):
    """
    Create a token to access the API.
    Only users belonging to the specified groups are allowed to create tokens for one or more of that groups.
    """
    username = claims.get("username")
    groups = claims.get("cognito:groups", [])
    # Maximum 90 days
    expires_in_days = min(90, body.expires_in_days)
    expire = datetime.now(timezone.utc) + timedelta(days=expires_in_days)
    allowed_groups = list(set(groups) & set(body.grouplist))
    if not allowed_groups:
        raise HTTPException(status_code=403, detail="User does not belong to any of the groups they want to access the API.")

    to_encode = {
        "sub": username,
        "groups": allowed_groups,
        "exp": expire,
        "iat": datetime.now(timezone.utc)
    }
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    
    return {
        "access_token": encoded_jwt,
        "token_type": "bearer",
        "expires_in_days": expires_in_days
    }

@app.get("/")
def read_root():
    """Health check endpoint."""
    return {
        "message": "FM Inference Service is running.",
        "version": "1.0.0",
        "api_version": "v1",
        "endpoints": {
            "inferences": f"{root_path}/v1/inferences",
            "finetuned_models": f"{root_path}/v1/models",
            "preloaded_events": f"{root_path}/v1/preloaded_events",
            "docs": f"{root_path}/docs",
            "health": f"{root_path}/health"
        }
    }


@app.get("/health")
def health_check():
    """Health check endpoint (legacy)."""
    return {"successCode": 200, "status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
