import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Literal

from core.config import get_settings
from core.memory_service import MemoryService


# Pydantic models
class ScanResponse(BaseModel):
    total_photos: int
    mem0_stored: int
    zep_stored: int
    message: str


class ResetResponse(BaseModel):
    mem0: str
    zep: str


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Search query text")
    provider: Literal["mem0", "zep"] = Field(..., description="Memory provider to search")


class SearchResult(BaseModel):
    content: str
    score: float
    image_url: Optional[str] = None
    metadata: dict = {}


# Global memory service instance
memory_service: Optional[MemoryService] = None


def get_app_base_url(request: Request) -> str:
    """Construct the base URL from the request."""
    # Get the scheme from headers (for proxy situations) or fallback to request.url
    scheme = request.headers.get("x-forwarded-proto", request.url.scheme)
    host = request.headers.get("x-forwarded-host", request.url.netloc)
    return f"{scheme}://{host}"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    global memory_service

    # Ensure photos directory exists
    settings = get_settings()
    photos_dir = Path(settings.photos_dir)
    photos_dir.mkdir(exist_ok=True)

    # Initialize memory service with a placeholder URL
    # It will use the request URL for generating image URLs
    memory_service = MemoryService(base_url="http://localhost:8000")

    yield

    # Cleanup if needed
    pass


# Create FastAPI app
app = FastAPI(
    title=get_settings().app_name,
    description="Evaluation tool for comparing Mem0 and Zep memory providers",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
# Allow all origins in development, but specify your frontend domain in production
allowed_origins = [
    "https://kun-dehuang.github.io",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8080",
    "http://127.0.0.1:8080",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

# 如果环境变量中有配置的额外源，也添加进去
import os
extra_origins = os.getenv("ALLOWED_ORIGINS", "")
if extra_origins:
    allowed_origins.extend([origin.strip() for origin in extra_origins.split(",")])

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Routes
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "app": get_settings().app_name,
        "version": "1.0.0",
        "endpoints": {
            "scan": "POST /api/scan",
            "reset": "DELETE /api/reset",
            "search": "POST /api/search",
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/api/scan", response_model=ScanResponse)
async def scan_photos(request: Request):
    """
    Scan the photos directory and generate memories in both Mem0 and Zep.

    This will:
    1. Clear existing memories
    2. Scan the photos/ directory for images
    3. Extract EXIF and GPS metadata
    4. Store memories in both Mem0 and Zep
    """
    global memory_service

    # Update the base URL based on current request
    base_url = get_app_base_url(request)
    memory_service.base_url = base_url.rstrip("/")

    try:
        # First reset to clear old data
        await memory_service.reset_memories()

        # Then scan and store new memories
        result = await memory_service.scan_and_store_memories()

        return ScanResponse(
            total_photos=result["total_photos"],
            mem0_stored=result["mem0_stored"],
            zep_stored=result["zep_stored"],
            message=f"Processed {result['total_photos']} photos"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scan failed: {str(e)}")


@app.delete("/api/reset", response_model=ResetResponse)
async def reset_memories(request: Request):
    """
    Clear all memories from both Mem0 and Zep.
    """
    global memory_service

    # Update the base URL
    base_url = get_app_base_url(request)
    memory_service.base_url = base_url.rstrip("/")

    try:
        result = await memory_service.reset_memories()
        return ResetResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")


@app.post("/api/search", response_model=list[SearchResult])
async def search_memories(request: SearchRequest, http_request: Request):
    """
    Search memories using the specified provider.

    Returns a list of matching memories with image URLs for display.
    """
    global memory_service

    # Update the base URL
    base_url = get_app_base_url(http_request)
    memory_service.base_url = base_url.rstrip("/")

    try:
        results = await memory_service.search_memories(
            query=request.query,
            provider=request.provider
        )
        return [SearchResult(**r) for r in results]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


# Static files for photos and identity
settings = get_settings()
photos_path = Path(settings.photos_dir)
identity_path = Path(settings.identity_dir)

# Ensure directories exist
photos_path.mkdir(exist_ok=True)
identity_path.mkdir(exist_ok=True)

# Mount photos directory
if photos_path.exists():
    app.mount("/static", StaticFiles(directory=str(photos_path)), name="static")

# Mount identity directory separately for self.jpg access
if identity_path.exists():
    app.mount("/identity", StaticFiles(directory=str(identity_path)), name="identity")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
