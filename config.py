"""Application configuration loaded from environment variables."""

import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Central config - reads from .env file."""

    # LM Studio (local LLM)
    LM_STUDIO_BASE_URL: str = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
    LM_STUDIO_API_KEY: str = os.getenv("LM_STUDIO_API_KEY", "lm-studio")
    LM_STUDIO_MODEL: str = os.getenv("LM_STUDIO_MODEL", "qwen/qwen3-vl-4b")

    # Jenkins
    JENKINS_URL: str = os.getenv("JENKINS_URL", "http://localhost:8080")
    JENKINS_USER: str = os.getenv("JENKINS_USER", "admin")
    JENKINS_API_TOKEN: str = os.getenv("JENKINS_API_TOKEN", "")

    # App
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))

    # Monitor
    MONITOR_ENABLED: bool = os.getenv("MONITOR_ENABLED", "true").lower() == "true"
    MONITOR_INTERVAL: int = int(os.getenv("MONITOR_INTERVAL", "30"))

settings = Settings()
