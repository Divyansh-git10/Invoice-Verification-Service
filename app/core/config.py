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

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
    )


settings = Settings()
