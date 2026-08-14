from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APP_NAME: str = "Invoice Verification Service"
    APP_VERSION: str = "1.0.0"

    MAX_FILE_SIZE_MB: int = 10

    SUPPORTED_MIME_TYPES: tuple[str, ...] = (
        "application/pdf",
        "image/jpeg",
        "image/png",
    )

    # LLM escalation fallback (Groq). Absent key -> deterministic-only pipeline.
    GROQ_API_KEY: Optional[str] = None
    GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"
    LLM_MODEL: str = "llama-3.1-8b-instant"
    LLM_TIMEOUT_SECONDS: float = 5.0
    # Minimum LLM confidence required to override the deterministic winner.
    LLM_OVERRIDE_MIN_CONFIDENCE: float = 0.7

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
    )


settings = Settings()
