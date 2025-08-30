from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os


from .api.v1 import (
    inference_router,
    models_router,
    preloaded_events_router
)

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


root_path = os.environ.get("FASTAPI_ROOT_PATH", "")

app = FastAPI(
    title="FM Inference Service",
    description="REST API for managing finetuned models and inferences.",
    version="0.0.1",
    lifespan=lifespan,
    root_path=os.environ.get("FASTAPI_ROOT_PATH", "")
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include v1 API routers
app.include_router(inference_router)
app.include_router(models_router)
app.include_router(preloaded_events_router)


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
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
