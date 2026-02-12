from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # API Keys
    gemini_api_key: str = ""
    mem0_api_key: str = ""
    zep_api_key: str = ""

    # Mem0 Configuration
    mem0_api_base: str = "https://api.mem0.ai"
    mem0_user_id: str = "memory-test-user"

    # Zep Configuration
    zep_api_base: str = ""
    zep_session_id: str = "memory-test-session"

    # App Configuration
    photos_dir: str = "photos"
    identity_dir: str = "identity"
    app_name: str = "Memory Evaluation Tool"

    # Gemini Configuration
    gemini_model: str = "gemini-1.5-flash"
    gemini_analysis_prompt: str = ""

    # Protagonist Configuration
    protagonist_tag: str = "【主角】"
    protagonist_self_pronouns: list[str] = ["我", "我的", "我也要", "我在"]

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache
def get_settings():
    return Settings()
