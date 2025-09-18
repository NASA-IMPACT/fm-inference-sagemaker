from fastapi import FastAPI, Request, Depends, status, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from contextlib import asynccontextmanager
import os
from datetime import datetime, timezone, timedelta
from jose import JWTError, jwt
import logging
from typing import Any, Optional
from pydantic import BaseModel, Field
import base64
import json
import boto3
import hmac
import hashlib
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
COGNITO_USER_POOL_ID = os.environ.get("COGNITO_USER_POOL_ID")
COGNITO_CLIENT_SECRET = os.environ.get("COGNITO_CLIENT_SECRET")
COGNITO_CLIENT_ID = os.environ.get("COGNITO_CLIENT_ID")
AWS_REGION = os.environ.get("AWS_REGION", "us-west-2")
cognito_client = boto3.client('cognito-idp', region_name=AWS_REGION)

async def require_alb_authentication(request: Request) -> dict[str, Any]:
    """Dependency to ensure a user is authenticated by the ALB."""
    access_token = request.headers.get("x-amzn-oidc-accesstoken")
    if not access_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User is not authenticated via ALB/Cognito."
        )
    return get_jwt_payload(access_token)


def get_jwt_payload(token: str) -> dict[str, Any]:
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


def get_user_groups_from_cognito(username: str) -> list[str]:
    """Get user's groups from Cognito User Pool."""
    try:
        if not COGNITO_USER_POOL_ID:
            logger.error("COGNITO_USER_POOL_ID not set, cannot fetch groups")
            return []
            
        client = boto3.client('cognito-idp', region_name=AWS_REGION)
        
        logger.info(f"Fetching groups for user: {username} from pool: {COGNITO_USER_POOL_ID}")
        
        response = client.admin_list_groups_for_user(
            UserPoolId=COGNITO_USER_POOL_ID,
            Username=username
        )
        
        groups = [group['GroupName'] for group in response.get('Groups', [])]
        logger.info(f"User {username} belongs to groups: {groups}")
        return groups
        
    except Exception as e:
        logger.error(f"Error fetching user groups from Cognito: {type(e).__name__}: {str(e)}")
        return []

async def verify_cognito_token(
    creds: Optional[HTTPAuthorizationCredentials] = Depends(oauth2_scheme)
) -> Optional[dict[str, Any]]:
    """Dependency to validate Cognito access tokens directly."""
    if not creds:
        return None
    
    token = creds.credentials
    try:
        client = boto3.client('cognito-idp', region_name=AWS_REGION)
        response = client.get_user(AccessToken=token)
        
        username = response['Username']
        groups = get_user_groups_from_cognito(username)
        
        logger.info(f"Successfully authenticated user '{username}' via Cognito token")
        
        return {
            "sub": username,
            "username": username,
            "groups": groups,
            "cognito:groups": groups,
            "auth_method": "cognito_direct"
        }
    except Exception as e:
        logger.debug(f"Token is not a valid Cognito token: {e}")
        return None



async def verify_custom_token(
    creds: Optional[HTTPAuthorizationCredentials] = Depends(oauth2_scheme)
) -> Optional[dict[str, Any]]:
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
    request: Request,
    custom_token_payload: Optional[dict[str, Any]] = Depends(verify_custom_token),
    cognito_token_payload: Optional[dict[str, Any]] = Depends(verify_cognito_token)

) -> dict[str, Any]:
    """General authentication dependency that doesn't require a specific group."""
    if custom_token_payload:
        logger.info(f"Authenticating via custom JWT for user '{custom_token_payload.get('sub')}'.")
        return custom_token_payload
    if cognito_token_payload:
        logger.info(f"Authenticating via Cognito token for user '{cognito_token_payload.get('username')}'.")
        return cognito_token_payload
    logger.info("No valid JWT or Cognito token found. Falling back to Header authentication.")
    try:
        alb_claims = await require_alb_authentication(request)
        logger.info(f"User '{alb_claims.get('username')}' authorized via Headers.")
        return alb_claims
    except HTTPException as e:
        logger.warning("Header authentication failed.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated. Provide a valid bearer token (custom JWT or Cognito) or authenticate via ALB.",
        ) from e

# Include v1 API routers
inference_router = create_inference_router(general_access_dependency)
app.include_router(inference_router)
app.include_router(models_router)
app.include_router(preloaded_events_router)


@app.post("/create-token", tags=["Authentication"])
async def create_token(
    body: TokenRequest,
    claims: dict[str, Any] = Depends(require_alb_authentication),
    
):
    """
    Create a token to access the API.
    Only users belonging to the specified groups are allowed to create tokens for one or more of that groups.
    """
    username = claims.get("username")
    email = claims.get("email")
    groups = claims.get("cognito:groups", [])
    # Maximum 90 days
    expires_in_days = min(90, body.expires_in_days)
    expire = datetime.now(timezone.utc) + timedelta(days=expires_in_days)
    allowed_groups = list(set(groups) & set(body.grouplist))
    if not allowed_groups:
        raise HTTPException(status_code=403, detail="The user does not belong to any of the groups required to access the API.")

    to_encode = {
        "sub": username,
        "groups": allowed_groups,
        "email": email,
        "exp": expire,
        "iat": datetime.now(timezone.utc)
    }
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    
    return {
        "access_token": encoded_jwt,
        "token_type": "bearer",
        "expires_in_days": expires_in_days
    }


class LoginRequest(BaseModel):
    username: str
    password: str


def get_secret_hash(username: str, client_id: str, client_secret: str) -> str:
    """Generate secret hash for Cognito authentication (only needed if client has secret)"""
    message = username + client_id
    dig = hmac.new(
        client_secret.encode('utf-8'),
        message.encode('utf-8'),
        hashlib.sha256
    ).digest()
    return base64.b64encode(dig).decode()

async def authenticate_with_cognito(username: str, password: str) -> dict:
    """Authenticate user with Cognito and return user attributes"""
    try:
        auth_params = {
            'USERNAME': username,
            'PASSWORD': password,
        }
        
        # Add SECRET_HASH if your app client has a secret
        if COGNITO_CLIENT_SECRET:
            auth_params['SECRET_HASH'] = get_secret_hash(username, COGNITO_CLIENT_ID, COGNITO_CLIENT_SECRET)
        
        # Use USER_PASSWORD_AUTH instead of USER_SRP_AUTH
        auth_response = cognito_client.initiate_auth(
            ClientId=COGNITO_CLIENT_ID,
            AuthFlow='USER_PASSWORD_AUTH',
            AuthParameters=auth_params
        )
        
        # Check if authentication completed or if there's a challenge
        if 'ChallengeName' in auth_response:
            raise HTTPException(
                status_code=400, 
                detail=f"Authentication challenge required: {auth_response['ChallengeName']}"
            )
        
        # Extract access token from auth response
        access_token = auth_response['AuthenticationResult']['AccessToken']
        
        # Get user info using the access token (more efficient than admin_get_user)
        user_response = cognito_client.get_user(AccessToken=access_token)
        
        # Extract user attributes
        user_attributes = {}
        for attr in user_response['UserAttributes']:
            user_attributes[attr['Name']] = attr['Value']
        
        # Get user groups
        groups_response = cognito_client.admin_list_groups_for_user(
            UserPoolId=COGNITO_USER_POOL_ID,
            Username=username
        )
        
        groups = [group['GroupName'] for group in groups_response['Groups']]
        
        return {
            'username': username,
            'email': user_attributes.get('email'),
            'cognito:groups': groups,
            'user_attributes': user_attributes,
            'access_token': access_token,  # Include the Cognito access token if needed
            'id_token': auth_response['AuthenticationResult'].get('IdToken'),
            'refresh_token': auth_response['AuthenticationResult'].get('RefreshToken')
        }
        
    except cognito_client.exceptions.NotAuthorizedException:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    except cognito_client.exceptions.UserNotFoundException:
        raise HTTPException(status_code=401, detail="User not found")
    except cognito_client.exceptions.UserNotConfirmedException:
        raise HTTPException(status_code=401, detail="User account not confirmed")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Authentication failed: {str(e)}")

def create_jwt_token(user_data: dict, expire_days: int = 7) -> dict:
    """Create JWT token from user data"""
    expire = datetime.now(timezone.utc) + timedelta(days=expire_days)
    
    to_encode = {
        "sub": user_data["username"],
        "groups": user_data.get("cognito:groups", []),
        "exp": expire,
        "email": user_data.get("email"),
        "iat": datetime.now(timezone.utc)
    }
    
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    
    return {
        "access_token": encoded_jwt,
        "token_type": "bearer",
        "expires_in_days": expire_days
    }

# New endpoint for username/password authentication
@app.post("/login", tags=["Authentication"])
async def login_with_credentials(login_request: LoginRequest):
    """
    Authenticate with username and password to get a JWT token.
    """
    user_data = await authenticate_with_cognito(
        login_request.username, 
        login_request.password
    )
    
    return create_jwt_token(user_data)

# Your existing ALB authentication endpoint (modified to use the helper function)
@app.post("/token", tags=["Authentication"])
async def get_token(
    claims: dict[str, Any] = Depends(require_alb_authentication)
):
    """
    Get a JWT for any authenticated user via ALB. The token will contain
    the user's group memberships and can be used to authenticate subsequent requests.
    """
    return create_jwt_token(claims)

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
