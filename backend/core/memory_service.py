"""
Vision Memory Lab - Enhanced Memory Service
==================================================
Features:
1. Identity Anchor: Face encoding extraction from identity/self.jpg
2. Visual Analysis Pipeline: VLM + Face Recognition + EXIF parsing
3. Dual Storage: Mem0 + Zep v2 API
4. Query Rewrite: Map "我" to "【主角】"
"""
import asyncio
import io
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, asdict

import httpx
from PIL import Image
import pillow_heif
from pillow_heif import register_heif_opener
import google.generativeai as genai

# Face recognition imports
try:
    import face_recognition
    import numpy as np
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    face_recognition = None
    np = None
    print("Warning: face_recognition not available, face matching disabled")

# EXIF parsing imports
try:
    import exifread
    EXIFREAD_AVAILABLE = True
except ImportError:
    EXIFREAD_AVAILABLE = False
    exifread = None
    print("Warning: exifread not available, EXIF data extraction disabled")

from .config import get_settings

# Register HEIF opener globally
register_heif_opener()


@dataclass
class EXIFData:
    """Container for EXIF metadata."""
    gps_latitude: Optional[float] = None
    gps_longitude: Optional[float] = None
    datetime_original: Optional[str] = None
    orientation: Optional[int] = None
    camera_make: Optional[str] = None
    camera_model: Optional[str] = None
    lens_model: Optional[str] = None
    focal_length: Optional[str] = None
    iso: Optional[int] = None
    aperture: Optional[str] = None
    shutter_speed: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class FaceMatchResult:
    """Result of face matching analysis."""
    has_protagonist: bool = False
    face_count: int = 0
    protagonist_distances: List[float] = field(default_factory=list)
    best_distance: Optional[float] = None


@dataclass
class PhotoAnalysis:
    """Complete analysis result for a photo."""
    description: str
    has_protagonist: bool
    face_count: int
    exif: Optional[EXIFData] = None
    file_path: str = ""
    filename: str = ""
    width: int = 0
    height: int = 0
    format: str = ""

    def to_metadata(self) -> Dict[str, Any]:
        """Convert to metadata dict for storage."""
        metadata = {
            "has_protagonist": self.has_protagonist,
            "face_count": self.face_count,
            "filename": self.filename,
            "file_path": self.file_path,
            "width": self.width,
            "height": self.height,
            "format": self.format,
            "source": "vision_analysis"
        }
        if self.exif:
            metadata["exif"] = self.exif.to_dict()
        return metadata


class EXIFParser:
    """EXIF metadata parser using exifread."""

    # Face recognition threshold (lower = more strict)
    # Typical values: 0.6 (strict) to 0.5 (moderate)
    FACE_MATCH_THRESHOLD = 0.5

    @staticmethod
    def _convert_to_degrees(value) -> float:
        """Convert EXIF GPS coordinates to degrees."""
        d = float(value.values[0].num) / float(value.values[0].den)
        m = float(value.values[1].num) / float(value.values[1].den)
        s = float(value.values[2].num) / float(value.values[2].den)
        return d + (m / 60.0) + (s / 3600.0)

    @classmethod
    def parse(cls, image_path: Path) -> EXIFData:
        """Parse EXIF data from image file."""
        if not EXIFREAD_AVAILABLE:
            return EXIFData()

        try:
            with open(image_path, 'rb') as f:
                tags = exifread.process_file(f, details=False)

            exif = EXIFData()

            # GPS coordinates
            if 'GPS GPSLatitude' in tags and 'GPS GPSLatitudeRef' in tags:
                lat = cls._convert_to_degrees(tags['GPS GPSLatitude'])
                if str(tags['GPS GPSLatitudeRef']) == 'S':
                    lat = -lat
                exif.gps_latitude = lat

            if 'GPS GPSLongitude' in tags and 'GPS GPSLongitudeRef' in tags:
                lon = cls._convert_to_degrees(tags['GPS GPSLongitude'])
                if str(tags['GPS GPSLongitudeRef']) == 'W':
                    lon = -lon
                exif.gps_longitude = lon

            # Date/time
            if 'EXIF DateTimeOriginal' in tags:
                exif.datetime_original = str(tags['EXIF DateTimeOriginal'])

            # Orientation
            if 'Image Orientation' in tags:
                exif.orientation = int(tags['Image Orientation'].values[0])

            # Camera info
            if 'Image Make' in tags:
                exif.camera_make = str(tags['Image Make'])
            if 'Image Model' in tags:
                exif.camera_model = str(tags['Image Model'])
            if 'EXIF LensModel' in tags:
                exif.lens_model = str(tags['EXIF LensModel'])

            # Shooting settings
            if 'EXIF FocalLength' in tags:
                f_val = tags['EXIF FocalLength']
                exif.focal_length = f"{f_val.num / f_val.den:.1f}mm"
            if 'EXIF ISOSpeedRatings' in tags:
                iso_tag = tags['EXIF ISOSpeedRatings']
                # Handle both numeric values and string representations
                if hasattr(iso_tag, 'values'):
                    exif.iso = int(iso_tag.values[0])
                else:
                    try:
                        exif.iso = int(iso_tag)
                    except (ValueError, TypeError):
                        exif.iso = None
            if 'EXIF FNumber' in tags:
                f_val = tags['EXIF FNumber']
                exif.aperture = f"f/{f_val.num / f_val.den:.1f}"
            if 'EXIF ExposureTime' in tags:
                e_val = tags['EXIF ExposureTime']
                exif.shutter_speed = f"{e_val.num}/{e_val.den}s"

            return exif

        except Exception as e:
            print(f"EXIF parsing error for {image_path.name}: {e}")
            return EXIFData()


class FaceMatcher:
    """Face recognition matcher using face_recognition library."""

    # Known face encoding cache
    _protagonist_encoding: Optional[List] = None

    @classmethod
    def load_protagonist_encoding(cls, identity_path: Path) -> Optional[List]:
        """Extract and cache face encoding from identity photo."""
        if not FACE_RECOGNITION_AVAILABLE:
            return None

        if not identity_path.exists():
            print(f"Identity photo not found: {identity_path}")
            return None

        try:
            # Load image and detect faces
            image = face_recognition.load_image_file(str(identity_path))
            encodings = face_recognition.face_encodings(image)

            if not encodings:
                print(f"No face detected in identity photo: {identity_path}")
                return None

            if len(encodings) > 1:
                print(f"Multiple faces detected in identity photo, using first face")

            cls._protagonist_encoding = encodings[0]
            print(f"Protagonist face encoding extracted from {identity_path.name}")
            return cls._protagonist_encoding

        except Exception as e:
            print(f"Failed to extract protagonist face encoding: {e}")
            return None

    @classmethod
    def match_faces(cls, image_path: Path, threshold: float = None) -> FaceMatchResult:
        """Detect and match faces in image against protagonist encoding."""
        if not FACE_RECOGNITION_AVAILABLE:
            return FaceMatchResult(face_count=0)

        if cls._protagonist_encoding is None:
            return FaceMatchResult(face_count=0)

        if threshold is None:
            threshold = cls.FACE_MATCH_THRESHOLD

        try:
            # Load image and detect faces
            image = face_recognition.load_image_file(str(image_path))
            face_locations = face_recognition.face_locations(image)
            face_encodings = face_recognition.face_encodings(image, face_locations)

            result = FaceMatchResult(face_count=len(face_locations))
            result.protagonist_distances = []

            # Compare each detected face with protagonist
            for encoding in face_encodings:
                distance = face_recognition.face_distance([cls._protagonist_encoding], encoding)[0]
                result.protagonist_distances.append(float(distance))

                if distance <= threshold:
                    result.has_protagonist = True

            if result.protagonist_distances:
                result.best_distance = min(result.protagonist_distances)

            return result

        except Exception as e:
            print(f"Error matching faces in {image_path.name}: {e}")
            return FaceMatchResult(face_count=0)

    @classmethod
    def clear_cache(cls):
        """Clear cached encoding."""
        cls._protagonist_encoding = None


class ZepAdapter:
    """Zep v3 HTTP API client - thread-based memory storage."""

    # Rate limit retry configuration
    MAX_RETRIES = 3
    RETRY_DELAY_BASE = 20  # Base delay in seconds

    def __init__(self):
        self.settings = get_settings()
        self.api_key = self.settings.zep_api_key
        self.base_url = self.settings.zep_api_base or "https://api.getzep.com"
        self.user_id = self.settings.mem0_user_id
        # V3 uses threads instead of sessions
        self.thread_id = self.settings.zep_session_id

    def _get_headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Api-Key {self.api_key}"
        }

    async def _retry_with_backoff(self, func, *args, **kwargs):
        """Retry with exponential backoff for rate limiting (410)."""
        for attempt in range(self.MAX_RETRIES):
            try:
                return await func(*args, **kwargs)
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 410:
                    # Rate limited - wait and retry
                    delay = self.RETRY_DELAY_BASE * (2 ** attempt)
                    print(f"Zep rate limited (410), retry {attempt + 1}/{self.MAX_RETRIES} after {delay}s")
                    await asyncio.sleep(delay)
                else:
                    # Other HTTP error - re-raise
                    raise
            except Exception as e:
                # Non-HTTP error - re-raise on last attempt or raise immediately
                if attempt == self.MAX_RETRIES - 1:
                    raise
                print(f"Zep request failed (attempt {attempt + 1}): {e}")

        # All retries exhausted
        raise Exception(f"Zep API: Max retries ({self.MAX_RETRIES}) exceeded")

    async def _ensure_user_thread(self) -> bool:
        """Ensure user and thread exist."""
        if not self.api_key:
            return False

        return await self._retry_with_backoff(
            self._ensure_user_thread_impl
        )

    async def _ensure_user_thread_impl(self) -> bool:
        """Actual implementation of user/thread creation."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Create user - use POST to /api/v2/users with user_id in body
                user_url = f"{self.base_url.rstrip('/')}/api/v2/users"
                user_payload = {"user_id": self.user_id}
                user_response = await client.post(user_url, headers=self._get_headers(), json=user_payload)
                print(f"Zep user creation: {user_response.status_code}")

                # Create thread - use POST to /api/v2/threads with thread_id in body
                thread_url = f"{self.base_url.rstrip('/')}/api/v2/threads"
                thread_payload = {"thread_id": self.thread_id, "user_id": self.user_id}
                thread_response = await client.post(thread_url, headers=self._get_headers(), json=thread_payload)
                print(f"Zep thread creation: {thread_response.status_code}")

                # 409 means thread already exists, which is fine
                if thread_response.status_code not in (200, 201, 409):
                    print(f"Zep thread creation error: {thread_response.text}")

                return thread_response.status_code in (200, 201, 409)
        except httpx.HTTPStatusError as e:
            print(f"HTTP error ensuring Zep user/thread: {e.response.status_code} - {e.response.text}")
            return False
        except Exception as e:
            print(f"Error ensuring Zep user/thread: {e}")
            return False

    async def add_memory(self, content: str, metadata: Dict[str, Any] = None) -> bool:
        """Add memory to Zep thread as a message with retry."""
        if not self.api_key:
            return False

        return await self._retry_with_backoff(
            self._add_memory_impl,
            content, metadata
        )

    async def _add_memory_impl(self, content: str, metadata: Dict[str, Any] = None) -> bool:
        """Actual implementation of adding memory."""
        # Ensure user and thread exist first
        await self._ensure_user_thread()

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # V3 API: POST to /threads/{thread_id}/messages
                # Note: endpoint expects "messages" array with a single message
                url = f"{self.base_url.rstrip('/')}/api/v2/threads/{self.thread_id}/messages"
                message = {
                    "role": "user",
                    "content": content
                }
                if metadata:
                    message["metadata"] = metadata

                payload = {"messages": [message]}

                response = await client.post(url, headers=self._get_headers(), json=payload)
                print(f"Zep add response: {response.status_code}")
                if response.status_code not in (200, 201):
                    print(f"Zep add error response: {response.text}")
                return response.status_code in (200, 201)
        except Exception as e:
            print(f"Zep add error: {e}")
            return False

    async def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search memories in Zep thread with retry."""
        if not self.api_key:
            return []

        return await self._retry_with_backoff(
            self._search_impl,
            query, limit
        )

    async def _search_impl(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Actual implementation of searching memories."""
        print(f"[DEBUG] ZepAdapter._search_impl: query='{query}', limit={limit}")
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # V3 API: Get messages from thread, then filter for query
                # Note: Zep does not support direct search on thread messages
                # We retrieve messages and can do client-side filtering
                url = f"{self.base_url.rstrip('/')}/api/v2/threads/{self.thread_id}/messages"
                print(f"[DEBUG] Fetching from: {url}")
                response = await client.get(url, headers=self._get_headers())

                print(f"[DEBUG] Zep response status: {response.status_code}")
                if response.status_code == 200:
                    data = response.json()
                    print(f"[DEBUG] Zep data type: {type(data)}")

                    # Zep returns: {"messages": [...], "total_count": N, "row_count": N, "user_id": "..."}
                    if isinstance(data, dict) and "messages" in data:
                        results = data["messages"]
                    elif isinstance(data, list):
                        results = data
                    else:
                        print(f"Zep returned unexpected structure: {data}")
                        return []

                    print(f"[DEBUG] Zep raw results count: {len(results)}")
                    formatted = []
                    for i, r in enumerate(results[:limit]):
                        # Zep returns messages with role, content, metadata, uuid, created_at, timestamp
                        if isinstance(r, dict):
                            content = r.get("content", "")
                            metadata = r.get("metadata", {})
                            formatted.append({
                                "content": content,
                                "score": 1.0,  # All messages from thread are relevant
                                "image_url": metadata.get("image_url", ""),
                                "metadata": metadata
                            })
                        else:
                            print(f"[DEBUG] Skipping non-dict result {i}: {type(r)}")
                    print(f"[DEBUG] Zep formatted results count: {len(formatted)}")
                    return formatted
                else:
                    print(f"Zep get messages failed: {response.status_code}")
                    return []
        except Exception as e:
            import traceback
            print(f"Zep search error: {e}")
            print(f"Traceback:\n{traceback.format_exc()}")
            return []

    async def reset(self) -> bool:
        """Reset by deleting and recreating thread with retry."""
        if not self.api_key:
            return False

        return await self._retry_with_backoff(
            self._reset_impl
        )

    async def _reset_impl(self) -> bool:
        """Actual implementation of reset."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Delete thread (V3 uses threads instead of sessions)
                url = f"{self.base_url.rstrip('/')}/api/v2/threads/{self.thread_id}"
                response = await client.delete(url, headers=self._get_headers())

                # Recreate thread (ignore delete result)
                await self._ensure_user_thread()
                return True
        except Exception as e:
            print(f"Zep reset error: {e}")
            return False


class Mem0Client:
    """Mem0 API client for memory storage."""

    def __init__(self):
        self.settings = get_settings()
        self.api_base = self.settings.mem0_api_base
        self.api_key = self.settings.mem0_api_key
        self.user_id = self.settings.mem0_user_id

    async def add(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Add memory to Mem0."""
        if not self.api_key:
            return False

        try:
            async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
                response = await client.post(
                    f"{self.api_base}/v1/memories?user_id={self.user_id}",
                    headers={"Authorization": f"Token {self.api_key}"},
                    json={
                        "messages": [
                            {"role": "user", "content": content, "metadata": metadata or {}}
                        ]
                    }
                )
                print(f"Mem0 add response: {response.status_code}")
                return response.status_code in (200, 201)
        except Exception as e:
            print(f"Mem0 add error: {e}")
            return False

    async def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search Mem0 memories."""
        if not self.api_key:
            return []

        try:
            async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
                response = await client.get(
                    f"{self.api_base}/v1/memories/search?user_id={self.user_id}&query={query}&limit={limit}",
                    headers={"Authorization": f"Token {self.api_key}"}
                )

                if response.status_code == 200:
                    results = response.json()
                    if isinstance(results, list):
                        formatted = []
                        for r in results:
                            if isinstance(r, dict):
                                # Ensure score is float
                                if 'score' in r and r['score'] is not None:
                                    r['score'] = float(r['score'])
                                formatted.append(r)
                        return formatted
                    return results.get("results", [])
                else:
                    return results.get("results", [])
        except Exception as e:
            print(f"Mem0 search error: {e}")
            return []

    async def delete_all(self) -> bool:
        """Delete all memories for user."""
        if not self.api_key:
            return False

        try:
            async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
                response = await client.delete(
                    f"{self.api_base}/v1/memories?user_id={self.user_id}",
                    headers={"Authorization": f"Token {self.api_key}"}
                )
                print(f"Mem0 delete response: {response.status_code}")
                return response.status_code in (200, 204)
        except Exception as e:
            print(f"Mem0 delete error: {e}")
            return False


class GeminiVisionAnalyzer:
    """Gemini 1.5 Flash VLM for image analysis."""

    MODELS = [
        "models/gemini-2.5-flash",
        "models/gemini-2.0-flash",
        "models/gemini-flash-latest",
    ]

    def __init__(self):
        self.settings = get_settings()
        genai.configure(api_key=self.settings.gemini_api_key)

        # Initialize model with fallback
        self.model = None
        self.model_name = None

        model_map = {
            "gemini-1.5-flash": "models/gemini-2.5-flash",
            "gemini-1.5-pro": "models/gemini-2.0-flash",
            "gemini-pro-vision": "models/gemini-2.5-flash",
        }

        configured = self.settings.gemini_model
        candidates = [model_map.get(configured, configured)] + self.MODELS

        for candidate in candidates:
            try:
                self.model = genai.GenerativeModel(candidate)
                self.model_name = candidate
                print(f"Using Gemini model: {candidate}")
                break
            except Exception:
                    continue

        if self.model is None:
            raise RuntimeError("No valid Gemini model found")

        self._executor = ThreadPoolExecutor(max_workers=4)
        self.protagonist_features: Optional[str] = None

    def _load_image_bytes(self, image_path: Path, max_size: Tuple[int, int] = (1024, 1024)) -> bytes:
        """Load and resize image for API transmission."""
        with Image.open(image_path) as img:
            # Convert RGB/RGBA
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Resize to reduce payload
            img.thumbnail(max_size, Image.Resampling.LANCZOS)

            # Convert to bytes
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='JPEG', quality=85)
            img_bytes.seek(0)
            return img_bytes.getvalue()

    async def extract_protagonist_features(self, identity_path: Path) -> Optional[str]:
        """Extract semantic description of protagonist from identity photo."""
        if not identity_path.exists():
            print("Identity photo not found")
            return None

        loop = asyncio.get_event_loop()

        def _analyze():
            try:
                img_bytes = self._load_image_bytes(identity_path)

                prompt = """
                请详细描述这张照片中人物的外貌特征，用于后续主角识别。
                包括：
                1. 性别和年龄段
                2. 发型（颜色、长度、风格）
                3. 面部特征（眼镜、脸型等）
                4. 体型
                5. 常见服装风格
                6. 其他显著特征（纹身、配饰等）
                """

                response = self.model.generate_content([
                    prompt,
                    {"mime_type": "image/jpeg", "data": img_bytes}
                ])

                return response.text
            except Exception as e:
                print(f"Failed to extract protagonist features: {e}")
                return None

        self.protagonist_features = await loop.run_in_executor(self._executor, _analyze)

        return self.protagonist_features

    def _build_analysis_prompt(self, face_match: FaceMatchResult) -> str:
        """Build VLM prompt with face match context."""
        base_prompt = self.settings.gemini_analysis_prompt

        # Add face match context
        face_context = ""
        if face_match.has_protagonist:
            face_context = "\n**人脸识别结果**: 已检测到主角在照片中（基于生物特征比对）。"
        elif face_match.face_count > 0:
            face_context = f"\n**人脸识别结果**: 检测到 {face_match.face_count} 个人脸，但未匹配到主角。"
        else:
            face_context = "\n**人脸识别结果**: 未检测到人脸。"

        # Add protagonist features if available
        if self.protagonist_features:
            return f'''【主角参考特征】:
{self.protagonist_features}

{face_context}

{base_prompt}
            '''
        else:
            return f'''
{face_context}

{base_prompt}
            '''

    async def analyze_photo(
        self,
        image_path: Path,
        face_match: FaceMatchResult
    ) -> Optional[PhotoAnalysis]:
        """Analyze photo with VLM and extract metadata."""
        if not image_path.exists():
            return None

        loop = asyncio.get_event_loop()

        def _analyze():
            try:
                # Get image dimensions
                with Image.open(image_path) as img:
                    width, height = img.size
                    fmt = img.format or "Unknown"

                # Load image for VLM
                img_bytes = self._load_image_bytes(image_path)

                # Build prompt with face context
                prompt = self._build_analysis_prompt(face_match)

                response = self.model.generate_content([
                    prompt,
                    {"mime_type": "image/jpeg", "data": img_bytes}
                ])

                description = response.text

                # Check if protagonist tag is in description
                has_protagonist_vlm = self.settings.protagonist_tag in description

                # Combine face match and VLM detection
                # If face recognition says yes, trust it; otherwise check VLM
                has_protagonist = face_match.has_protagonist or (
                    not FACE_RECOGNITION_AVAILABLE and has_protagonist_vlm
                )

                return PhotoAnalysis(
                    description=description,
                    has_protagonist=has_protagonist,
                    face_count=face_match.face_count,
                    file_path=str(image_path),
                    filename=image_path.name,
                    width=width,
                    height=height,
                    format=fmt
                )

            except Exception as e:
                print(f"Failed to analyze {image_path.name}: {e}")
                return None

        return await loop.run_in_executor(self._executor, _analyze)

    async def analyze_photos_batch(
        self,
        image_paths: List[Path],
        face_matches: List[FaceMatchResult],
        concurrency: int = 5
    ) -> List[Optional[PhotoAnalysis]]:
        """Analyze multiple photos with concurrency control."""
        semaphore = asyncio.Semaphore(concurrency)

        async def _analyze_with_semaphore(img_path: Path, face_match: FaceMatchResult):
            async with semaphore:
                return await self.analyze_photo(img_path, face_match)

        tasks = [
            _analyze_with_semaphore(p, fm)
            for p, fm in zip(image_paths, face_matches)
        ]
        return await asyncio.gather(*tasks)


class VisionMemoryLab:
    """
    Vision Memory Lab - Main Service
    ==================================
    1. Identity Anchor: Extract face encoding from identity/self.jpg
    2. Visual Analysis: VLM + Face Recognition + EXIF
    3. Dual Storage: Mem0 + Zep
    4. Query Rewrite: Map "我" to "【主角】"
    """

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.settings = get_settings()
        self.base_url = base_url.rstrip("/")
        self.zep = ZepAdapter()
        self.mem0 = Mem0Client()
        self.gemini = GeminiVisionAnalyzer()

        # Identity cache
        self._identity_loaded = False
        self._identity_path = Path(self.settings.identity_dir) / "self.jpg"

    async def initialize(self) -> bool:
        """Initialize Zep thread on startup."""
        try:
            return await self.zep._ensure_user_thread()
        except Exception as e:
            print(f"Failed to initialize Zep thread: {e}")
            return False

    async def init_identity(self) -> Dict[str, Any]:
        """
        Initialize identity anchor.
        - Extract semantic features via VLM
        - Extract face encoding via face_recognition
        """
        result = {
            "vlm_features": None,
            "face_encoding": False,
            "identity_path": str(self._identity_path)
        }

        # Extract VLM semantic features
        features = await self.gemini.extract_protagonist_features(self._identity_path)
        result["vlm_features"] = features is not None

        # Extract face encoding
        if FACE_RECOGNITION_AVAILABLE:
            encoding = FaceMatcher.load_protagonist_encoding(self._identity_path)
            result["face_encoding"] = encoding is not None

        self._identity_loaded = True
        return result

    def _rewrite_query(self, query: str) -> str:
        """Rewrite query: map pronouns to protagonist tag."""
        q = query
        for pronoun in self.settings.protagonist_self_pronouns:
            q = q.replace(pronoun, self.settings.protagonist_tag)
        return q

    async def add_memory(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        img_url: str = ""
    ) -> Dict[str, bool]:
        """Add memory to both Zep and Mem0."""
        # Add image_url to metadata
        if metadata is None:
            metadata = {}
        if img_url:
            metadata["image_url"] = img_url

        # Parallel add to both providers
        results = await asyncio.gather(
            self.zep.add_memory(content, metadata),
            self.mem0.add(content, metadata),
            return_exceptions=True
        )

        return {
            "zep": results[0] if not isinstance(results[0], Exception) else False,
            "mem0": results[1] if not isinstance(results[1], Exception) else False
        }

    async def reset(self) -> Dict[str, str]:
        """Reset all memories."""
        zep_ok = await self.zep.reset()
        mem0_ok = await self.mem0.delete_all()

        return {
            "zep": "cleared" if zep_ok else "error",
            "mem0": "cleared" if mem0_ok else "error"
        }

    async def sync_photos(
        self,
        force_refresh: bool = False
    ) -> Dict[str, Any]:
        """
        Sync photos: Analyze and store memories.

        Pipeline for each photo:
        1. Parse EXIF metadata
        2. Match faces against protagonist
        3. VLM analysis with context
        4. Store to dual providers
        """
        photos_dir = Path(self.settings.photos_dir)

        # Find all image files
        image_files = []
        for ext in ["*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.png", "*.PNG",
                    "*.heic", "*.HEIC", "*.webp", "*.WEBP"]:
            image_files.extend(photos_dir.glob(ext))

        if not image_files:
            return {
                "total_photos": 0,
                "newly_added": 0,
                "zep_stored": 0,
                "mem0_stored": 0,
                "protagonist_photos": 0,
                "face_recognition": FACE_RECOGNITION_AVAILABLE
            }

        # Reset if force refresh
        if force_refresh:
            await self.reset()

        # Ensure identity is loaded
        if not self._identity_loaded:
            await self.init_identity()

        # Step 1: Parse EXIF and match faces (I/O bound, can be parallel)
        exif_results: List[EXIFData] = []
        face_results: List[FaceMatchResult] = []

        # Process in batches for memory efficiency
        batch_size = 50
        for i in range(0, len(image_files), batch_size):
            batch = image_files[i:i + batch_size]

            for img_path in batch:
                # EXIF parsing
                exif = EXIFParser.parse(img_path)
                exif_results.append(exif)
                # Face matching
                face_match = FaceMatcher.match_faces(img_path)
                face_results.append(face_match)

            print(f"Processed batch {i // batch_size + 1}/{(len(image_files) + batch_size - 1) // batch_size}")

        # Step 2: VLM analysis (concurrent API calls)
        analyses = await self.gemini.analyze_photos_batch(
            image_files,
            face_results,
            concurrency=5
        )

        # Step 3: Store results with EXIF data
        zep_stored = 0
        mem0_stored = 0
        protagonist_count = 0

        for img_path, analysis, exif in zip(image_files, analyses, exif_results):
            if analysis is None:
                continue

            # Attach EXIF data
            analysis.exif = exif
            # Generate image URL
            img_url = f"{self.base_url}/static/{img_path.name}"

            # Store
            result = await self.add_memory(
                analysis.description,
                analysis.to_metadata(),
                img_url
            )

            if result.get("zep"):
                zep_stored += 1
            if result.get("mem0"):
                mem0_stored += 1
            if analysis.has_protagonist:
                protagonist_count += 1

        return {
            "total_photos": len(image_files),
            "newly_added": len(image_files) if force_refresh else 0,
            "zep_stored": zep_stored,
            "mem0_stored": mem0_stored,
            "protagonist_photos": protagonist_count,
            "face_recognition": FACE_RECOGNITION_AVAILABLE
        }

    async def search(
        self,
        query: str,
        provider: str = "zep",
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search memories by provider."""
        rewritten = self._rewrite_query(query)

        if provider == "zep":
            results = await self.zep.search(rewritten, limit)
        elif provider == "mem0":
            results = await self.mem0.search(rewritten, limit)
        else:
            results = []

        # Filter results by relevance - match query against content
        if query and results:
            results = self._filter_by_relevance(results, query, limit)

        # Ensure image_url in results
        for r in results:
            metadata = r.get("metadata", {})
            if "image_url" not in r:
                r["image_url"] = metadata.get("image_url", "")
            r["metadata"] = metadata

        return results

    def _filter_by_relevance(self, results: List[Dict[str, Any]], query: str, limit: int) -> List[Dict[str, Any]]:
        """Filter results by relevance based on query matching."""
        if not results:
            return []

        print(f"[DEBUG] _filter_by_relevance: query='{query}', limit={limit}, results_count={len(results)}")

        # Calculate relevance score for each result
        query_lower = query.lower()
        for i, r in enumerate(results):
            content = r.get("content", "")
            score = r.get("score", 0)

            # Debug: check types of existing score
            if score is not None:
                print(f"[DEBUG] Result {i}: existing score={score}, type={type(score)}")

            if content:
                content_lower = content.lower()
                # Simple relevance: check if query appears in content
                # For Chinese, check character overlap; for English, check word overlap
                if self._is_chinese(query_lower):
                    # Chinese: check character overlap
                    match_count = sum(1 for c in query_lower if c in content_lower and c.strip())
                    relevance = min(match_count / len(query_lower), 1.0) if query_lower else 0.5
                else:
                    # English: check word overlap
                    query_words = set(query_lower.split())
                    content_words = set(content_lower.split())
                    if query_words:
                        overlap = len(query_words & content_words) / len(query_words)
                        relevance = overlap
                    else:
                        relevance = 0.5  # No query words, give neutral score

                r["score"] = float(relevance)  # Ensure score is always a float
                print(f"[DEBUG] Result {i}: calculated relevance={relevance}")
            else:
                r["score"] = 0.0
                print(f"[DEBUG] Result {i}: no content, score=0.0")

        # Sort by score (relevance) and return top results
        try:
            sorted_results = sorted(results, key=lambda x: float(x.get("score", 0)), reverse=True)
            return sorted_results[:limit]
        except TypeError as e:
            print(f"[ERROR] Sort failed: {e}")
            print(f"[DEBUG] Results scores: {[r.get('score') for r in results]}")
            # Fallback: return results without sorting
            return results[:limit]

    def _is_chinese(self, text: str) -> bool:
        """Check if text contains Chinese characters."""
        return any(0x4e00 <= ord(c) <= 0x9fff for c in text)


# Legacy MemoryService alias for compatibility
MemoryService = VisionMemoryLab
