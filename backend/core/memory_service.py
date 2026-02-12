import os
import io
import asyncio
import base64
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor

import google.generativeai as genai
from PIL import Image, ExifTags
from PIL.Image import Exif

# Register HEIC opener
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass  # pillow-heif not installed, HEIC support not available

from mem0 import Memory
from zep_cloud import Zep

from .config import get_settings


# Default Visual Anthropologist Prompt
DEFAULT_ANALYSIS_PROMPT = """你是一位专业的视觉人类学家，负责分析照片并提取关键信息。

请仔细观察这张照片，按照以下结构进行分析：

0. 主角识别：根据之前提供的主角特征描述，判断照片中是否出现主角。如果出现，请在分析开头标注【主角】。
1. 场景环境：描述照片的拍摄环境（室内/室外、地点类型、光线条件、背景元素等）。
2. 人物互动：描述照片中的人物活动、互动方式、情绪状态等。
3. 物品消费：识别照片中的物品、食物、饮料、商品等，并描述其类型和状态。
4. 动作活动：描述照片中的主要动作、事件或活动内容。
5. 视觉风格：描述照片的构图、色彩、氛围等视觉特征。

请以自然流畅的段落形式输出分析结果，如果检测到主角出现，务必在开头明确标注【主角】。"""


class ProtagonistFeatures:
    """Store protagonist identity features extracted from identity/self.jpg."""

    def __init__(self):
        self.features: str = ""
        self.loaded: bool = False


# Global protagonist features storage
self_protagonist_features = ProtagonistFeatures()


class GeminiAnalyzer:
    """Gemini 1.5 Flash client for image analysis."""

    def __init__(self):
        self.settings = get_settings()
        genai.configure(api_key=self.settings.gemini_api_key)
        self.model = genai.GenerativeModel(self.settings.gemini_model)
        self.executor = ThreadPoolExecutor(max_workers=4)

    def _load_image_bytes(self, image_path: str) -> bytes:
        """Load image as bytes for Gemini API."""
        with open(image_path, "rb") as f:
            return f.read()

    async def extract_protagonist_features(self, image_path: str) -> str:
        """
        Extract protagonist appearance features from identity/self.jpg.

        Returns detailed description of protagonist's appearance including
        hair, glasses, body type, clothing style, and other distinguishing features.
        """
        prompt = """请详细描述这张照片中人物的外貌特征，包括：

1. 发型：发色、发型、头发长度等
2. 面部特征：是否戴眼镜、面部轮廓、胡须等
3. 体型：身高体型估计
4. 服装风格：穿着习惯、风格偏好
5. 其他显著特征：纹身、饰品、特殊标记等

请用简洁但详细的语言描述，用于在后续照片中识别此人。"""

        try:
            image_bytes = self._load_image_bytes(image_path)
            image = Image.from_bytes(image_bytes)

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                self.executor,
                lambda: self.model.generate_content([prompt, image])
            )

            features = response.text.strip()
            self_protagonist_features.features = features
            self_protagonist_features.loaded = True
            return features

        except Exception as e:
            print(f"Error extracting protagonist features: {e}")
            return ""

    def get_analysis_prompt(self) -> str:
        """Get the analysis prompt from env or default."""
        custom_prompt = self.settings.gemini_analysis_prompt
        if custom_prompt and custom_prompt.strip():
            return custom_prompt
        return DEFAULT_ANALYSIS_PROMPT

    def build_image_prompt(self, image_path: str) -> str:
        """Build the full prompt for image analysis."""
        base_prompt = self.get_analysis_prompt()

        if self_protagonist_features.loaded and self_protagonist_features.features:
            protagonist_info = f"""
【主角特征参考】：
{self_protagonist_features.features}

请根据以上特征识别照片中是否出现主角。
"""
            return protagonist_info + "\n" + base_prompt
        else:
            # No protagonist features loaded, skip protagonist detection
            return base_prompt

    async def analyze_photo(self, image_path: str) -> Dict[str, Any]:
        """
        Analyze a photo using Gemini 1.5 Flash.

        Returns:
            Dict with keys:
            - analysis: str - Full analysis text
            - has_protagonist: bool - Whether protagonist is detected
            - scene: str - Scene description
            - error: str - Error message if analysis failed
        """
        prompt = self.build_image_prompt(image_path)

        try:
            image_bytes = self._load_image_bytes(image_path)
            image = Image.from_bytes(image_bytes)

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                self.executor,
                lambda: self.model.generate_content([prompt, image])
            )

            analysis = response.text.strip()

            # Check if protagonist tag is present
            protagonist_tag = self.settings.protagonist_tag
            has_protagonist = protagonist_tag in analysis

            # Extract structured info
            result = {
                "analysis": analysis,
                "has_protagonist": has_protagonist,
                "protagonist_detected": has_protagonist,
                "scene": self._extract_scene_info(analysis),
                "error": None
            }

            return result

        except Exception as e:
            return {
                "analysis": "",
                "has_protagonist": False,
                "protagonist_detected": False,
                "scene": "",
                "error": str(e)
            }

    def _extract_scene_info(self, analysis: str) -> str:
        """Extract scene information from analysis."""
        # Simple extraction - take first 100 chars as scene summary
        if not analysis:
            return ""

        # Remove protagonist tag for scene summary
        clean_text = analysis.replace(self.settings.protagonist_tag, "").strip()

        # Return first sentence or first 150 chars
        sentences = clean_text.split("。")
        if sentences:
            return sentences[0].strip()[:150]

        return clean_text[:150]


class ImageMetadata:
    """Extract and store image metadata."""

    @staticmethod
    def get_exif_data(image_path: str) -> Dict[str, Any]:
        """Extract EXIF data from image."""
        try:
            with Image.open(image_path) as img:
                exif_data = {}
                if hasattr(img, '_getexif') and img._getexif() is not None:
                    exif = img._getexif()
                    for tag, value in exif.items():
                        decoded = ExifTags.TAGS.get(tag, tag)
                        exif_data[decoded] = str(value)
                return exif_data
        except Exception as e:
            return {"error": str(e)}

    @staticmethod
    def get_gps_data(image_path: str) -> Optional[Dict[str, float]]:
        """Extract GPS coordinates from image EXIF."""
        try:
            with Image.open(image_path) as img:
                if not hasattr(img, '_getexif') or img._getexif() is None:
                    return None

                exif = img._getexif()
                gps_info = None
                for tag, value in exif.items():
                    decoded = ExifTags.TAGS.get(tag, tag)
                    if decoded == "GPSInfo":
                        gps_info = value
                        break

                if not gps_info:
                    return None

                def convert_to_degrees(value):
                    """Convert GPS coordinates to degrees."""
                    d, m, s = value
                    return d + (m / 60.0) + (s / 3600.0)

                def get_gps_value(exif, key):
                    """Get GPS value from EXIF data."""
                    for k, v in exif.items():
                        decoded = ExifTags.GPSTAGS.get(k, k)
                        if decoded == key:
                            return v
                    return None

                lat = get_gps_value(gps_info, "GPSLatitude")
                lat_ref = get_gps_value(gps_info, "GPSLatitudeRef")
                lon = get_gps_value(gps_info, "GPSLongitude")
                lon_ref = get_gps_value(gps_info, "GPSLongitudeRef")

                if lat and lon and lat_ref and lon_ref:
                    lat = convert_to_degrees(lat)
                    lon = convert_to_degrees(lon)
                    if lat_ref == "S":
                        lat = -lat
                    if lon_ref == "W":
                        lon = -lon
                    return {"latitude": lat, "longitude": lon}
                return None
        except Exception:
            return None

    @staticmethod
    def get_image_info(image_path: str) -> Dict[str, Any]:
        """Get comprehensive image information."""
        try:
            with Image.open(image_path) as img:
                info = {
                    "filename": os.path.basename(image_path),
                    "format": img.format,
                    "mode": img.mode,
                    "size": img.size,
                    "width": img.width,
                    "height": img.height,
                }

                # Add GPS data if available
                gps = ImageMetadata.get_gps_data(image_path)
                if gps:
                    info["gps"] = gps

                # Add basic EXIF
                exif = ImageMetadata.get_exif_data(image_path)
                if exif and "error" not in exif:
                    info["exif"] = exif

                return info
        except Exception as e:
            return {"filename": os.path.basename(image_path), "error": str(e)}


class MemoryService:
    """Unified memory service supporting both Mem0 and Zep with Gemini analysis."""

    def __init__(self, base_url: str):
        self.settings = get_settings()
        self.base_url = base_url.rstrip("/")
        self.photos_dir = Path(self.settings.photos_dir)
        self.identity_dir = Path(self.settings.identity_dir)

        # Initialize Gemini analyzer
        if self.settings.gemini_api_key:
            self.gemini = GeminiAnalyzer()
        else:
            self.gemini = None
            print("Warning: GEMINI_API_KEY not set, image analysis disabled")

        # Initialize Mem0
        if self.settings.mem0_api_key:
            self.mem0_client = Memory.from_api(
                api_key=self.settings.mem0_api_key,
                user_id=self.settings.mem0_user_id
            )
        else:
            self.mem0_client = None
            print("Warning: MEM0_API_KEY not set")

        # Initialize Zep Cloud
        if self.settings.zep_api_key:
            self.zep_client = Zep(
                api_key=self.settings.zep_api_key
            )
        else:
            self.zep_client = None
            print("Warning: ZEP_API_KEY not set")

    def _get_static_url(self, filename: str) -> str:
        """Generate full static URL for an image."""
        return f"{self.base_url}/static/{filename}"

    def _preprocess_query(self, query: str) -> str:
        """
        Preprocess search query by replacing pronouns with protagonist tag.
        Replaces "我", "我的", "我也要", "我在" with "【主角】".
        """
        processed = query
        for pronoun in self.settings.protagonist_self_pronouns:
            processed = processed.replace(pronoun, self.settings.protagonist_tag)
        return processed

    def _scan_photos_directory(self) -> List[Dict[str, Any]]:
        """Scan photos directory and extract metadata."""
        if not self.photos_dir.exists():
            return []

        photos = []
        # Support HEIC format via pillow-heif if available, otherwise standard formats
        image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp', '.tiff', '.heic', '.heif'}

        for file_path in sorted(self.photos_dir.rglob('*')):
            if file_path.suffix.lower() in image_extensions:
                info = ImageMetadata.get_image_info(str(file_path))

                photos.append({
                    "filename": info['filename'],
                    "path": str(file_path),
                    "static_url": self._get_static_url(info['filename']),
                    "metadata": info
                })

        return photos

    async def initialize_protagonist(self) -> bool:
        """
        Initialize protagonist features from identity/self.jpg.

        Returns:
            bool: True if protagonist features were successfully extracted
        """
        if not self.gemini:
            print("Gemini not configured, skipping protagonist initialization")
            return False

        self_path = self.identity_dir / "self.jpg"
        self_path_jpg = self.identity_dir / "self.JPG"

        # Find self.jpg file (case insensitive)
        for path in [self_path, self_path_jpg]:
            if path.exists():
                try:
                    features = await self.gemini.extract_protagonist_features(str(path))
                    print(f"Protagonist features extracted: {features[:100]}...")
                    return True
                except Exception as e:
                    print(f"Failed to extract protagonist features: {e}")
                    return False

        print("No identity/self.jpg found, protagonist detection disabled")
        return False

    async def _analyze_photo_concurrent(self, photos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Analyze photos concurrently using Gemini with error handling.

        Args:
            photos: List of photo dicts with 'path' key

        Returns:
            List of photo dicts with added 'analysis' key
        """
        if not self.gemini:
            # Return photos with empty analysis
            for photo in photos:
                photo["analysis"] = {
                    "analysis": "",
                    "has_protagonist": False,
                    "protagonist_detected": False,
                    "scene": "",
                    "error": "Gemini not configured"
                }
            return photos

        # Create analysis tasks for all photos
        async def analyze_single(photo: Dict[str, Any]) -> Dict[str, Any]:
            try:
                result = await self.gemini.analyze_photo(photo["path"])
                photo["analysis"] = result
                return photo
            except Exception as e:
                photo["analysis"] = {
                    "analysis": "",
                    "has_protagonist": False,
                    "protagonist_detected": False,
                    "scene": "",
                    "error": str(e)
                }
                return photo

        # Run all analyses concurrently
        tasks = [analyze_single(photo) for photo in photos]
        results = await asyncio.gather(*tasks, return_exceptions=False)

        return results

    async def scan_and_store_memories(self) -> Dict[str, Any]:
        """
        Scan photos directory and store memories in both Mem0 and Zep.

        Process:
        1. Initialize protagonist features from identity/self.jpg
        2. Ensure Zep user and session exist
        3. Scan photos directory
        4. Analyze each photo with Gemini concurrently
        5. Store results in both Mem0 and Zep
        """
        # Step 1: Initialize protagonist
        await self.initialize_protagonist()

        # Step 2: Ensure Zep user and session exist
        if self.zep_client:
            try:
                await self.zep_client.user.add(user_id=self.settings.mem0_user_id)
            except Exception:
                pass  # User might already exist

            try:
                await self.zep_client.session.add(
                    session_id=self.settings.zep_session_id,
                    user_id=self.settings.mem0_user_id,
                )
            except Exception:
                pass  # Session might already exist

        # Step 3: Scan photos
        photos = self._scan_photos_directory()

        if not photos:
            return {
                "total_photos": 0,
                "mem0_stored": 0,
                "zep_stored": 0,
                "photos": []
            }

        # Step 4: Analyze photos concurrently
        photos = await self._analyze_photo_concurrent(photos)

        # Step 5: Store in memory providers
        mem0_stored = 0
        zep_stored = 0

        for photo in photos:
            analysis = photo.get("analysis", {})
            metadata = photo["metadata"]
            has_protagonist = analysis.get("has_protagonist", False)
            scene = analysis.get("scene", "")
            full_analysis = analysis.get("analysis", "")

            # Build rich content for storage
            if full_analysis:
                content_for_mem0 = full_analysis
                content_for_zep = scene or full_analysis
            else:
                # Fallback to basic description if no analysis
                filename = metadata.get('filename', 'unknown')
                content_for_mem0 = f"Photo: {filename}"
                if metadata.get('gps'):
                    content_for_mem0 += f", taken at GPS: {metadata['gps']}"
                if metadata.get('width') and metadata.get('height'):
                    content_for_mem0 += f", dimensions: {metadata['width']}x{metadata['height']}"
                content_for_zep = content_for_mem0

            # Metadata for storage
            storage_metadata = {
                "filename": metadata.get('filename', ''),
                "image_url": photo.get('static_url', ''),
                "has_protagonist": has_protagonist,
                "scene": scene,
                **{k: v for k, v in metadata.items() if k not in ['filename', 'error']}
            }

            # Store in Mem0
            if self.mem0_client:
                try:
                    self.mem0_client.add(
                        message=content_for_mem0,
                        metadata=storage_metadata,
                        user_id=self.settings.mem0_user_id
                    )
                    mem0_stored += 1
                except Exception as e:
                    print(f"Mem0 storage error for {metadata.get('filename')}: {e}")

            # Store in Zep Cloud
            if self.zep_client:
                try:
                    # Zep Cloud uses message format for memory
                    from zep_cloud import Message

                    await self.zep_client.memory.add(
                        session_id=self.settings.zep_session_id,
                        messages=[
                            Message(
                                role="user",
                                content=content_for_zep,
                                metadata=storage_metadata
                            )
                        ]
                    )
                    zep_stored += 1
                except Exception as e:
                    print(f"Zep storage error for {metadata.get('filename')}: {e}")

        return {
            "total_photos": len(photos),
            "mem0_stored": mem0_stored,
            "zep_stored": zep_stored,
            "photos": photos
        }

    async def reset_memories(self) -> Dict[str, Any]:
        """Clear all memories from both providers."""
        results = {}

        # Clear Mem0
        if self.mem0_client:
            try:
                memories = self.mem0_client.get_all(user_id=self.settings.mem0_user_id)
                for memory in memories:
                    self.mem0_client.delete(memory['id'], user_id=self.settings.mem0_user_id)
                results["mem0"] = "cleared"
            except Exception as e:
                results["mem0"] = f"error: {str(e)}"
        else:
            results["mem0"] = "not configured"

        # Clear Zep
        if self.zep_client:
            try:
                # Delete and recreate session
                await self.zep_client.session.delete(
                    session_id=self.settings.zep_session_id
                )
                # Recreate user and session
                await self.zep_client.user.add(user_id=self.settings.mem0_user_id)
                await self.zep_client.session.add(
                    session_id=self.settings.zep_session_id,
                    user_id=self.settings.mem0_user_id,
                )
                results["zep"] = "cleared"
            except Exception as e:
                results["zep"] = f"error: {str(e)}"
        else:
            results["zep"] = "not configured"

        # Reset protagonist features
        self_protagonist_features.features = ""
        self_protagonist_features.loaded = False
        results["protagonist"] = "reset"

        return results

    async def search_memories(self, query: str, provider: str) -> List[Dict[str, Any]]:
        """
        Search memories using specified provider.

        Query preprocessing: Replaces "我" (I/me) with "【主角】" for better matching.
        """
        # Preprocess query
        processed_query = self._preprocess_query(query)

        if processed_query != query:
            print(f"Query preprocessed: '{query}' -> '{processed_query}'")

        if provider == "mem0":
            return await self._search_mem0(processed_query)
        elif provider == "zep":
            return await self._search_zep(processed_query)
        return []

    async def _search_mem0(self, query: str) -> List[Dict[str, Any]]:
        """Search memories using Mem0."""
        if not self.mem0_client:
            return [{"error": "Mem0 not configured"}]

        try:
            results = self.mem0_client.search(
                query=query,
                user_id=self.settings.mem0_user_id,
                limit=10
            )

            formatted = []
            for r in results:
                formatted.append({
                    "content": r.get('memory', ''),
                    "score": r.get('score', 0),
                    "image_url": r.get('metadata', {}).get('image_url', ''),
                    "metadata": r.get('metadata', {})
                })
            return formatted
        except Exception as e:
            return [{"error": str(e)}]

    async def _search_zep(self, query: str) -> List[Dict[str, Any]]:
        """Search memories using Zep Cloud."""
        if not self.zep_client:
            return [{"error": "Zep not configured"}]

        try:
            from zep_cloud import Message

            results = await self.zep_client.memory.search(
                session_id=self.settings.zep_session_id,
                text=query,
                limit=10
            )

            formatted = []
            for r in results:
                # Zep Cloud returns Message objects with content and metadata
                formatted.append({
                    "content": r.content or '',
                    "score": 1.0 - (r.dist or 0),  # Convert distance to similarity
                    "image_url": (r.metadata or {}).get('image_url', ''),
                    "metadata": r.metadata or {}
                })
            return formatted
        except Exception as e:
            return [{"error": str(e)}]
