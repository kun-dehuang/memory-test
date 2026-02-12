"""
Pure HTTP client for Zep Cloud API.
Fallback when zep_cloud package is not available.
"""
import os
import httpx
from typing import List, Dict, Any, Optional
from .config import get_settings


class ZepHttpClient:
    """
    Simple HTTP client wrapper for Zep Cloud API.
    """

    def __init__(self, base_url: str, api_key: str, session_id: str, user_id: str):
        self.settings = get_settings()
        self.api_key: Optional[str] = api_key
        self.base_url: Optional[str] = base_url
        self.session_id: str = session_id
        self.user_id: str = user_id

        if not self.api_key:
            print("Warning: ZEP_API_KEY not set")

        # Set default base URL if not provided
        if not self.base_url:
            self.base_url = "https://api.getzep.com"

        # Create HTTP client with timeout
        self.client = httpx.Client(
            base_url=self.base_url,
            headers={},
            timeout=30.0
        )

    def _get_headers(self) -> Dict[str, str]:
        """Get request headers with API key."""
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Api-Key {self.api_key}"
        return headers

    async def create_session(self) -> bool:
        """Create a new session for the user."""
        if not self.api_key:
            print("Cannot create session: ZEP_API_KEY not set")
            return False

        headers = self._get_headers()

        # First create user
        user_url = f"{self.base_url.rstrip('/')}/api/v2/users/{self.user_id}"
        try:
            response = self.client.post(user_url, headers=headers)
            if response.status_code == 200 or response.status_code == 201:
                print(f"User created: {self.user_id}")
                return True
            elif response.status_code == 409:
                # User already exists
                print(f"User already exists: {self.user_id}")
                return True
            else:
                print(f"Failed to create user: {response.status_code}")
                return False
        except Exception as e:
            print(f"Error creating user: {e}")
            return False

    async def add_memory(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Add a message/memory to a session."""
        if not self.api_key:
            print("Cannot add memory: ZEP_API_KEY not set")
            return False

        headers = self._get_headers()

        # Add message to session
        session_url = f"{self.base_url.rstrip('/')}/api/v2/sessions/{self.session_id}/messages"
        payload = {
            "role": role,
            "content": content
        }

        if metadata:
            payload["metadata"] = metadata

        try:
            response = self.client.post(session_url, headers=headers, json=payload)
            if response.status_code == 200 or response.status_code == 201:
                return True
            else:
                print(f"Failed to add memory: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            print(f"Error adding memory: {e}")
            return False

    async def search_memories(
        self,
        query: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search memories using Zep Cloud vector search."""
        if not self.api_key:
            return [{"error": "ZEP_API_KEY not set"}]

        headers = self._get_headers()

        # Use memory search endpoint
        search_url = f"{self.base_url.rstrip('/')}/api/v2/memory/search"
        payload = {
            "text": query,
            "limit": limit
        }

        try:
            response = self.client.post(search_url, headers=headers, json=payload)

            if response.status_code == 200:
                results = response.json()
                formatted = []
                for r in results.get("results", []):
                    # Extract content and metadata
                    content = r.get("content", "")
                    metadata = r.get("metadata", {})

                    # Format result
                    formatted.append({
                        "content": content,
                        "score": 1.0,  # Zep returns similarity score directly
                        "image_url": metadata.get("image_url", ""),
                        "metadata": metadata
                    })
                return formatted
            else:
                print(f"Search failed: {response.status_code}")
                return []
        except Exception as e:
            print(f"Search error: {e}")
            return [{"error": str(e)}]

    async def delete_session(self) -> bool:
        """Delete a session (all messages)."""
        if not self.api_key:
            print("Cannot delete session: ZEP_API_KEY not set")
            return False

        headers = self._get_headers()

        session_url = f"{self.base_url.rstrip('/')}/api/v2/sessions/{self.session_id}"
        try:
            response = self.client.delete(session_url, headers=headers)
            if response.status_code == 200 or response.status_code == 204:
                print(f"Session deleted: {self.session_id}")
                # Recreate empty session
                return await self.create_session()
            else:
                print(f"Failed to delete session: {response.status_code}")
                return False
        except Exception as e:
            print(f"Error deleting session: {e}")
            return False

    async def check_connection(self) -> bool:
        """Check if Zep API is accessible."""
        headers = self._get_headers()
        health_url = f"{self.base_url.rstrip('/')}/healthz"
        try:
            response = self.client.get(health_url, headers=headers, timeout=5.0)
            if response.status_code == 200 and response.text == ".":
                return True
        except Exception as e:
            print(f"Zep connection check failed: {e}")
            return False


class ZepMemoryService:
    """
    Pure HTTP client for Zep Cloud API.
    Compatible with existing MemoryService interface.
    """

    def __init__(self, base_url: str):
        self.settings = get_settings()
        self.base_url = base_url.rstrip("/")
        self.session_id: str = self.settings.zep_session_id
        self.user_id: str = self.settings.mem0_user_id
        self.client = ZepHttpClient(
            base_url=base_url,
            api_key=self.settings.zep_api_key,
            session_id=self.session_id,
            user_id=self.user_id
        )

    async def initialize(self) -> bool:
        """Initialize Zep session for the user."""
        return await self.client.create_session()

    async def add_memory(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Add a memory/message to the Zep session."""
        return await self.client.add_memory(role, content, metadata)

    async def search_memories(
        self,
        query: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search memories using Zep Cloud vector search."""
        return await self.client.search_memories(query, limit)

    async def reset_memories(self) -> Dict[str, Any]:
        """Clear all memories by deleting and recreating the session."""
        success = await self.client.delete_session()
        return {
            "zep": "cleared" if success else "error: Failed to clear memories"
        }

    async def check_connection(self) -> bool:
        """Check if Zep API is accessible."""
        return await self.client.check_connection()
