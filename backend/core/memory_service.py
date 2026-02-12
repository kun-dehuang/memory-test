import asyncio
from typing import Optional, Dict, Any, List
from pathlib import Path
import httpx

from .config import get_settings
from .zep_http_client import ZepHttpClient


class ZepAdapter:
    """Zep dual-mode adapter: SDK with HTTP fallback."""

    def __init__(self):
        self.settings = get_settings()
        self.sdk_client = None
        self.http_client: Optional[ZepHttpClient] = None
        self.mode = "none"

        # Try SDK first
        try:
            from zep_cloud.client import Zep as ZepClient
            self.sdk_client = ZepClient(api_key=self.settings.zep_api_key)
            self.mode = "sdk"
        except Exception as e:
            print(f"Zep SDK init failed: {e}, falling back to HTTP")
            self._init_http_client()

    def _init_http_client(self):
        """Initialize HTTP fallback client."""
        base_url = self.settings.zep_api_base or "https://api.getzep.com"
        self.http_client = ZepHttpClient(
            base_url=base_url,
            api_key=self.settings.zep_api_key,
            session_id=self.settings.zep_session_id,
            user_id=self.settings.mem0_user_id
        )
        self.mode = "http"

    async def add_memory(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Add memory via SDK or HTTP fallback."""
        if self.mode == "sdk" and self.sdk_client:
            try:
                self.sdk_client.memory.add(
                    session_id=self.settings.zep_session_id,
                    messages=[{"role": "user", "content": content}],
                    metadata=metadata or {}
                )
                return True
            except Exception as e:
                print(f"Zep SDK add failed: {e}, switching to HTTP")
                self._init_http_client()

        if self.http_client:
            return await self.http_client.add_memory("user", content, metadata)
        return False

    async def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search via SDK or HTTP fallback."""
        if self.mode == "sdk" and self.sdk_client:
            try:
                results = self.sdk_client.memory.search(
                    session_id=self.settings.zep_session_id,
                    text=query,
                    limit=limit
                )
                return [{"content": r.content, "score": r.dist, "metadata": r.metadata or {}} for r in results]
            except Exception as e:
                print(f"Zep SDK search failed: {e}, switching to HTTP")
                self._init_http_client()

        if self.http_client:
            return await self.http_client.search_memories(query, limit)
        return []

    async def initialize(self) -> bool:
        """Initialize Zep session."""
        if self.http_client:
            return await self.http_client.create_session()
        return True

    async def reset(self) -> bool:
        """Reset Zep memories."""
        if self.http_client:
            return await self.http_client.delete_session()
        return True


class Mem0Client:
    """Mem0 API client."""

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
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.api_base}/v1/memories",
                    headers={"Authorization": f"Token {self.api_key}"},
                    json={
                        "text": content,
                        "user_id": self.user_id,
                        "metadata": metadata or {}
                    },
                    timeout=30.0
                )
                return response.status_code in (200, 201)
        except Exception as e:
            print(f"Mem0 add failed: {e}")
            return False

    async def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search Mem0 memories."""
        if not self.api_key:
            return []

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.api_base}/v1/memories/search",
                    headers={"Authorization": f"Token {self.api_key}"},
                    json={
                        "query": query,
                        "user_id": self.user_id,
                        "limit": limit
                    },
                    timeout=30.0
                )
                if response.status_code == 200:
                    return response.json().get("results", [])
                return []
        except Exception as e:
            print(f"Mem0 search failed: {e}")
            return []


class MemoryService:
    """Unified memory service with Zep + Mem0 dual storage."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.settings = get_settings()
        self.base_url = base_url.rstrip("/")
        self.zep = ZepAdapter()
        self.mem0 = Mem0Client()

    async def initialize(self) -> Dict[str, bool]:
        """Initialize storage backends."""
        zep_ok = await self.zep.initialize()
        return {"zep": zep_ok, "mem0": bool(self.settings.mem0_api_key)}

    def _rewrite_query(self, query: str) -> str:
        """Rewrite query: map '我' to protagonist tag."""
        q = query
        for pronoun in self.settings.protagonist_self_pronouns:
            q = q.replace(pronoun, self.settings.protagonist_tag)
        return q

    async def add_memory(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, bool]:
        """Add memory to both Zep and Mem0."""
        results = await asyncio.gather(
            self.zep.add_memory(content, metadata),
            self.mem0.add(content, metadata),
            return_exceptions=True
        )
        return {
            "zep": results[0] if not isinstance(results[0], Exception) else False,
            "mem0": results[1] if not isinstance(results[1], Exception) else False
        }

    async def add_memories(self, memories: List[Dict[str, Any]]) -> Dict[str, int]:
        """Batch add memories."""
        success = {"zep": 0, "mem0": 0}
        tasks = []
        for m in memories:
            content = m.get("content", "")
            metadata = m.get("metadata", {})
            tasks.append(self.zep.add_memory(content, metadata))
            tasks.append(self.mem0.add(content, metadata))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for i, r in enumerate(results):
            if not isinstance(r, Exception) and r:
                if i % 2 == 0:
                    success["zep"] += 1
                else:
                    success["mem0"] += 1
        return success

    async def search(self, query: str, limit: int = 10) -> Dict[str, List[Dict[str, Any]]]:
        """Parallel search across Zep and Mem0."""
        rewritten = self._rewrite_query(query)
        results = await asyncio.gather(
            self.zep.search(rewritten, limit),
            self.mem0.search(rewritten, limit),
            return_exceptions=True
        )
        return {
            "zep": results[0] if not isinstance(results[0], Exception) else [],
            "mem0": results[1] if not isinstance(results[1], Exception) else [],
            "query_rewritten": rewritten != query
        }

    async def reset(self) -> Dict[str, str]:
        """Reset all memories."""
        zep_ok = await self.zep.reset()
        return {
            "zep": "cleared" if zep_ok else "error",
            "mem0": "not_implemented"
        }

    async def reset_memories(self) -> Dict[str, str]:
        """Reset all memories (alias for compatibility)."""
        return await self.reset()

    async def scan_and_store_memories(self) -> Dict[str, int]:
        """Scan photos directory and store memories."""
        photos_dir = Path(self.settings.photos_dir)
        image_files = list(photos_dir.glob("*.jpg")) + list(photos_dir.glob("*.png")) + list(photos_dir.glob("*.jpeg"))

        memories = []
        for img_path in image_files:
            img_url = f"{self.base_url}/static/{img_path.name}"
            content = f"Photo: {img_path.stem}"
            metadata = {"image_url": img_url, "filename": img_path.name}
            memories.append({"content": content, "metadata": metadata})

        result = await self.add_memories(memories)
        result["total_photos"] = len(image_files)
        return result

    async def search_memories(self, query: str, provider: str = "zep", limit: int = 10) -> List[Dict[str, Any]]:
        """Search memories by provider."""
        rewritten = self._rewrite_query(query)

        if provider == "zep":
            results = await self.zep.search(rewritten, limit)
        elif provider == "mem0":
            results = await self.mem0.search(rewritten, limit)
        else:
            results = []

        # Ensure image_url in metadata
        for r in results:
            if "image_url" not in r.get("metadata", {}):
                r.setdefault("metadata", {})["image_url"] = r.get("metadata", {}).get("image_url", "")
            if "image_url" not in r:
                r["image_url"] = r.get("metadata", {}).get("image_url", "")
        return results
