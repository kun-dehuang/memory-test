"""
Vision Memory Lab - FastAPI Backend
====================================
Endpoints:
- POST /api/init: Initialize identity anchor (VLM + Face Recognition)
- POST /api/sync: Sync photos with analysis pipeline
- POST /api/search: Search memories by provider
- DELETE /api/reset: Clear all memories
"""
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from core.config import get_settings
from core.memory_service import VisionMemoryLab, MemoryService


# =============================================================================
# Pydantic Models
# =============================================================================

class InitResponse(BaseModel):
    """Response for identity initialization."""
    vlm_features: bool
    face_encoding: bool
    identity_path: str
    message: str


class SyncResponse(BaseModel):
    """Response for photo sync operation."""
    total_photos: int
    newly_added: int
    zep_stored: int
    mem0_stored: int
    protagonist_photos: int
    face_recognition: bool
    message: str


class SearchRequest(BaseModel):
    """Search request body."""
    query: str = Field(..., min_length=1, description="Search query text")
    provider: Literal["mem0", "zep"] = Field(default="zep", description="Memory provider to search")
    limit: int = Field(default=10, ge=1, le=50, description="Maximum number of results")


class SearchResult(BaseModel):
    """Single search result."""
    content: str
    score: float
    image_url: Optional[str] = None
    metadata: dict = {}


class ResetResponse(BaseModel):
    """Response for reset operation."""
    zep: str
    mem0: str


# =============================================================================
# Global State
# =============================================================================

memory_service: Optional[MemoryService] = None


def get_app_base_url(request: Request) -> str:
    """Construct the base URL from the request."""
    scheme = request.headers.get("x-forwarded-proto", request.url.scheme)
    host = request.headers.get("x-forwarded-host", request.url.netloc)
    return f"{scheme}://{host}"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    global memory_service

    # Ensure required directories exist
    settings = get_settings()
    Path(settings.photos_dir).mkdir(exist_ok=True)
    Path(settings.identity_dir).mkdir(exist_ok=True)

    # Initialize memory service
    memory_service = MemoryService(base_url="http://localhost:8000")

    # Initialize Zep session
    try:
        await memory_service.initialize()
        print("Zep session initialized successfully")
    except Exception as e:
        print(f"Warning: Failed to initialize Zep session: {e}")

    yield

    # Cleanup if needed
    pass


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="Vision Memory Lab",
    description="Local evaluation tool for comparing Mem0 and Zep memory providers with VLM analysis",
    version="2.0.0",
    lifespan=lifespan
)

# Configure CORS
allowed_origins = [
    "https://kun-dehuang.github.io",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8080",
    "http://127.0.0.1:8080",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

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


# =============================================================================
# Routes
# =============================================================================

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "app": "Vision Memory Lab",
        "version": "2.0.0",
        "description": "Local evaluation tool for Mem0 vs Zep",
        "endpoints": {
            "init": "POST /api/init - Initialize identity anchor",
            "sync": "POST /api/sync - Sync and analyze photos",
            "search": "POST /api/search - Search memories",
            "reset": "DELETE /api/reset - Clear all memories",
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "service": "vision-memory-lab"}


@app.post("/api/init", response_model=InitResponse)
async def init_identity(request: Request):
    """
    Initialize identity anchor.

    This endpoint:
    1. Extracts semantic features from identity/self.jpg via VLM
    2. Extracts face encoding via face_recognition library

    Must be called before /api/sync for optimal protagonist detection.
    """
    global memory_service

    # Update base URL
    base_url = get_app_base_url(request)
    memory_service.base_url = base_url.rstrip("/")

    try:
        result = await memory_service.init_identity()

        status_parts = []
        if result["vlm_features"]:
            status_parts.append("VLM features extracted")
        if result["face_encoding"]:
            status_parts.append("Face encoding loaded")

        message = "Identity initialized. " + (", ".join(status_parts) if status_parts else "No features available")

        return InitResponse(
            vlm_features=result["vlm_features"] or False,
            face_encoding=result["face_encoding"],
            identity_path=result["identity_path"],
            message=message
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Init failed: {str(e)}")


@app.post("/api/sync", response_model=SyncResponse)
async def sync_photos(request: Request, force_refresh: bool = False):
    """
    Sync photos with full analysis pipeline.

    For each photo in the photos/ directory:
    1. Parse EXIF metadata (GPS, camera, shooting settings)
    2. Match faces against protagonist encoding
    3. VLM analysis with face match context
    4. Store to both Mem0 and Zep

    Query parameters:
    - force_refresh: If true, clears existing memories before syncing
    """
    global memory_service

    # Update base URL
    base_url = get_app_base_url(request)
    memory_service.base_url = base_url.rstrip("/")

    try:
        result = await memory_service.sync_photos(force_refresh=force_refresh)

        message = f"Processed {result['total_photos']} photos. "
        message += f"Zep: {result['zep_stored']}, Mem0: {result['mem0_stored']} stored. "
        message += f"Protagonist detected in {result['protagonist_photos']} photos."

        return SyncResponse(
            total_photos=result["total_photos"],
            newly_added=result["newly_added"],
            zep_stored=result["zep_stored"],
            mem0_stored=result["mem0_stored"],
            protagonist_photos=result["protagonist_photos"],
            face_recognition=result["face_recognition"],
            message=message
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sync failed: {str(e)}")


@app.post("/api/search", response_model=list[SearchResult])
async def search_memories(request: SearchRequest, http_request: Request):
    """
    Search memories using the specified provider.

    Request body:
    - query: Search query text (supports "我" -> "【主角】" rewrite)
    - provider: "zep" or "mem0"
    - limit: Number of results (1-50, default 10)
    """
    global memory_service

    print(f"[DEBUG] search_memories called: query='{request.query}', provider='{request.provider}', limit={request.limit} (type: {type(request.limit)})")

    # Update base URL
    base_url = get_app_base_url(http_request)
    memory_service.base_url = base_url.rstrip("/")

    try:
        results = await memory_service.search(
            query=request.query,
            provider=request.provider,
            limit=int(request.limit)  # Ensure limit is int
        )
        print(f"[DEBUG] Got {len(results)} results from memory_service.search")

        # Validate and convert results
        validated_results = []
        for i, r in enumerate(results):
            print(f"[DEBUG] Result {i}: keys={r.keys()}, score={r.get('score')} (type: {type(r.get('score'))})")
            # Ensure score is float
            if 'score' in r and r['score'] is not None:
                r['score'] = float(r['score'])
            validated_results.append(r)

        return [SearchResult(**r) for r in validated_results]
    except TypeError as e:
        import traceback
        print(f"[ERROR] TypeError in search: {e}")
        print(f"[ERROR] Traceback:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Search failed (type error): {str(e)}")
    except Exception as e:
        import traceback
        print(f"[ERROR] Exception in search: {e}")
        print(f"[ERROR] Traceback:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.delete("/api/reset", response_model=ResetResponse)
async def reset_memories(request: Request):
    """
    Clear all memories from both Mem0 and Zep.

    This will:
    1. Delete all documents from Zep collection
    2. Delete all memories from Mem0
    """
    global memory_service

    # Update base URL
    base_url = get_app_base_url(request)
    memory_service.base_url = base_url.rstrip("/")

    try:
        result = await memory_service.reset()
        return ResetResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")


# =============================================================================
# Static Files
# =============================================================================

settings = get_settings()
photos_path = Path(settings.photos_dir)
identity_path = Path(settings.identity_dir)

# Ensure directories exist
photos_path.mkdir(exist_ok=True)
identity_path.mkdir(exist_ok=True)

# Mount photos directory with HEIC/HEIF support
if photos_path.exists():
    from starlette.datastructures import UploadFile
    from starlette.responses import FileResponse
    import io
    import pillow_heif
    from PIL import Image
    from pathlib import Path as StdPath

    # Register HEIF opener
    pillow_heif.register_heif_opener()

    async def heic_image_response(file_path: str):
        """Serve HEIC images by converting to JPEG on the fly."""
        from starlette.responses import Response
        try:
            with Image.open(file_path) as img:
                # Convert to RGB and save as JPEG
                rgb_img = img.convert('RGB')
                img_io = io.BytesIO()
                rgb_img.save(img_io, format='JPEG', quality=85)
                img_io.seek(0)

                return Response(
                    content=img_io.getvalue(),
                    media_type="image/jpeg"
                )
        except Exception as e:
            print(f"Error converting HEIC: {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            # Fallback to original file
            return FileResponse(file_path)

    # Custom static file handler for HEIC support
    class HEICStaticFiles(StaticFiles):
        async def get_response(self, path: str, scope):
            # Check if this is a HEIC/HEIF file before delegating to parent
            file_path = StdPath(getattr(self, 'directory')) / path
            if file_path.exists() and file_path.suffix.lower() in ['.heic', '.heif']:
                return await heic_image_response(str(file_path))
            # For other files, use default StaticFiles behavior
            return await super().get_response(path, scope)

    app.mount("/static", HEICStaticFiles(directory=str(photos_path)), name="static")

# Mount identity directory
if identity_path.exists():
    app.mount("/identity", StaticFiles(directory=str(identity_path)), name="identity")


# =============================================================================
# Development Server
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
