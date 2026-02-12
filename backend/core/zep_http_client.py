"""
Pure HTTP client for Zep Cloud API.
Fallback when zep_cloud package is not available.
"""
import asyncio
import httpx
from typing import List, Dict, Any, Optional
from .config import get_settings


class ZepHttpClient:
    """
    Simple HTTP client wrapper for Zep Cloud API V3 (uses threads instead of threads).
    """

    def __init__(self, base_url: str, api_key: str, thread_id: str, user_id: str):
        self.settings = get_settings()
        self.api_key: Optional[str] = api_key
        self.base_url: Optional[str] = base_url
        self.thread_id: str = thread_id  # V3 uses threads
        self.user_id: str = user_id

        if not self.api_key:
            print("Warning: ZEP_API_KEY not set")

        # Set default base URL if not provided
        if not self.base_url:
            self.base_url = "https://api.getzep.com"

    def _get_headers(self) -> Dict[str, str]:
        """Get request headers with API key."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Api-Key {self.api_key}"
        return headers

    async def _retry_request(self, request_func, max_retries: int = 3):
        """Retry logic with exponential backoff for rate limiting."""
        last_error = None
        for attempt in range(max_retries):
            try:
                return await request_func()
            except httpx.HTTPStatusError as e:
                last_error = e
                if e.response.status_code == 410:
                    # Rate limited - wait and retry with exponential backoff
                    wait_time = (2 ** attempt) * 2
                    print(f"Rate limited (410), waiting {wait_time}s before retry {attempt + 1}/{max_retries}")
                    await asyncio.sleep(wait_time)
                    continue
                elif e.response.status_code == 429:
                    wait_time = (2 ** attempt) * 2
                    print(f"Too many requests (429), waiting {wait_time}s before retry {attempt + 1}/{max_retries}")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    raise
            except Exception as e:
                last_error = e
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(1)
        # All retries exhausted - raise the last error
        if last_error:
            raise last_error
        raise Exception(f"Max retries ({max_retries}) exceeded")

    async def create_thread(self) -> bool:
        """Create a new thread for the user."""
        if not self.api_key:
            print("Cannot create thread: ZEP_API_KEY not set")
            return False

        headers = self._get_headers()

        async def _create_user():
            user_url = f"{self.base_url.rstrip('/')}/api/v2/users/{self.user_id}"
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.put(user_url, headers=headers)
                response.raise_for_status()
                return True

        async def _create_thread():
            thread_url = f"{self.base_url.rstrip('/')}/api/v2/threads/{self.thread_id}"
            payload = {"user_id": self.user_id}
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(thread_url, headers=headers, json=payload)
                response.raise_for_status()
                return True

        try:
            # Create user first (idempotent - 409 means already exists)
            try:
                await self._retry_request(_create_user)
            except httpx.HTTPStatusError as e:
                if e.response.status_code not in (200, 201, 409):
                    raise

            # Create thread
            try:
                await self._retry_request(_create_thread)
            except httpx.HTTPStatusError as e:
                # 409 means thread already exists, which is fine
                if e.response.status_code not in (200, 201, 409):
                    raise

            print(f"Session ready: {self.thread_id} for user {self.user_id}")
            return True
        except Exception as e:
            print(f"Error creating thread: {e}")
            return False

    async def add_memory(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Add a message/memory to a thread."""
        if not self.api_key:
            print("Cannot add memory: ZEP_API_KEY not set")
            return False

        headers = self._get_headers()

        async def _request():
            # Add message to thread - use /messages endpoint
            # Note: endpoint expects "messages" array with a single message
            messages_url = f"{self.base_url.rstrip('/')}/api/v2/threads/{self.thread_id}/messages"
            message = {"role": role, "content": content}
            if metadata:
                message["metadata"] = metadata

            payload = {"messages": [message]}

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(messages_url, headers=headers, json=payload)
                response.raise_for_status()
                return True

        try:
            return await self._retry_request(_request) or True
        except Exception as e:
            print(f"Error adding memory: {e}")
            return False

    async def search_memories(
        self,
        query: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search memories using Zep Cloud vector search."""
        print(f"[DEBUG] search_memories called: query='{query}', limit={limit}")

        if not self.api_key:
            print("[DEBUG] No API key, returning empty")
            return []

        headers = self._get_headers()

        async def _request():
            # Get messages from thread - Zep doesn't support direct search on messages
            messages_url = f"{self.base_url.rstrip('/')}/api/v2/threads/{self.thread_id}/messages"
            print(f"[DEBUG] Fetching messages from: {messages_url}")

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(messages_url, headers=headers)
                print(f"[DEBUG] Zep response status: {response.status_code}")
                response.raise_for_status()
                data = response.json()
                print(f"[DEBUG] Zep response type: {type(data)}, keys: {data.keys() if isinstance(data, dict) else 'N/A'}")
                return data

        try:
            results = await self._retry_request(_request)
            if not results:
                print("[DEBUG] No results from Zep")
                return []

            # Handle different response structures
            messages = []
            if isinstance(results, list):
                messages = results
                print(f"[DEBUG] Results is a list with {len(messages)} items")
            elif isinstance(results, dict) and "messages" in results:
                messages = results["messages"]
                print(f"[DEBUG] Results is dict with 'messages' key: {len(messages)} items")
            elif isinstance(results, dict) and "results" in results:
                messages = results["results"]
                print(f"[DEBUG] Results is dict with 'results' key: {len(messages)} items")
            else:
                print(f"[DEBUG] Zep returned unexpected structure: {type(results)} = {results}")
                return []

            formatted = []
            for i, r in enumerate(messages[:limit]):
                print(f"[DEBUG] Processing message {i}: type={type(r)}, keys={r.keys() if isinstance(r, dict) else 'N/A'}")
                content = r.get("content", "")
                metadata = r.get("metadata", {})
                formatted.append({
                    "content": content,
                    "score": 1.0,  # All messages from thread are relevant
                    "image_url": metadata.get("image_url", ""),
                    "metadata": metadata
                })
            print(f"[DEBUG] Returning {len(formatted)} formatted results")
            return formatted
        except httpx.HTTPStatusError as e:
            print(f"[ERROR] Search failed: HTTP {e.response.status_code}")
            if e.response.status_code == 410:
                print("Zep API rate limit exceeded (410), please try again later")
            elif e.response.status_code == 401:
                print("Zep API authentication failed, check API key")
            elif e.response.status_code == 404:
                print(f"Thread not found: {self.thread_id}. Make sure the thread exists.")
            return []
        except Exception as e:
            import traceback
            print(f"[ERROR] Search error: {e}")
            print(f"[ERROR] Traceback:\n{traceback.format_exc()}")
            return []

    async def delete_thread(self) -> bool:
        """Delete a thread (all messages)."""
        if not self.api_key:
            print("Cannot delete thread: ZEP_API_KEY not set")
            return False

        headers = self._get_headers()

        async def _request():
            thread_url = f"{self.base_url.rstrip('/')}/api/v2/threads/{self.thread_id}"
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.delete(thread_url, headers=headers)
                response.raise_for_status()
                return True

        try:
            await self._retry_request(_request)
            print(f"Session deleted: {self.thread_id}")
            # Recreate empty thread
            return await self.create_thread()
        except Exception as e:
            print(f"Error deleting thread: {e}")
            return False

    async def check_connection(self) -> bool:
        """Check if Zep API is accessible."""
        headers = self._get_headers()
        health_url = f"{self.base_url.rstrip('/')}/healthz"
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(health_url, headers=headers)
                return response.status_code == 200
        except Exception:
            return False


class ZepMemoryService:
    """
    Pure HTTP client for Zep Cloud API.
    Compatible with existing MemoryService interface.
    """

    def __init__(self, base_url: str):
        self.settings = get_settings()
        self.base_url = base_url.rstrip("/")
        self.thread_id: str = self.settings.zep_session_id  # Keep using zep_session_id from config
        self.user_id: str = self.settings.mem0_user_id
        self.client = ZepHttpClient(
            base_url=base_url,
            api_key=self.settings.zep_api_key,
            thread_id=self.thread_id,
            user_id=self.user_id
        )

    async def initialize(self) -> bool:
        """Initialize Zep thread for the user."""
        return await self.client.create_thread()

    async def add_memory(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Add a memory/message to the Zep thread."""
        return await self.client.add_memory(role, content, metadata)

    async def search_memories(
        self,
        query: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search memories using Zep Cloud vector search."""
        return await self.client.search_memories(query, limit)

    async def reset_memories(self) -> Dict[str, Any]:
        """Clear all memories by deleting and recreating the thread."""
        success = await self.client.delete_thread()
        return {
            "zep": "cleared" if success else "error: Failed to clear memories"
        }

    async def check_connection(self) -> bool:
        """Check if Zep API is accessible."""
        return await self.client.check_connection()
