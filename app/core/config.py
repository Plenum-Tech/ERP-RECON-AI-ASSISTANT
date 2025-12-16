from functools import lru_cache

from pydantic import AnyHttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # App
    app_name: str = "Recon AI Assistant"
    environment: str = "local"

    # Database (async pg URL example: postgresql+asyncpg://user:pass@host/dbname)
    database_url: str

    # Upstream services (UC1 / UC2)
    uc1_base_url: AnyHttpUrl
    uc2_base_url: AnyHttpUrl

    http_timeout_seconds: int = 10

    # Pydantic v2 config
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache()
def get_settings() -> Settings:
    return Settings()
