# app/utils/settings.py

from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

try:
    # pydantic-settings v2
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ImportError:  # pragma: no cover
    # fallback for older pydantic
    from pydantic import BaseSettings  # type: ignore

    class SettingsConfigDict(dict):  # dummy so code still runs
        def __init__(self, **kwargs):
            super().__init__(**kwargs)


class AppSettings(BaseSettings):
    """
    Lightweight settings loader just for infra-type values
    that we don't want to fight with app.core.config for.

    It reads from `.env` by default and ignores unknown keys.
    """

    # This will be populated from either:
    # - `openai_api_key=...` in .env
    # - or `OPENAI_API_KEY=...` in .env / env
    openai_api_key: Optional[str] = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_app_settings() -> AppSettings:
    """
    Cached settings singleton.

    Uses pydantic-settings to load from `.env` and env vars.
    """
    return AppSettings()


def get_openai_api_key() -> Optional[str]:
    """
    Unified helper to retrieve the OpenAI API key.

    Priority:
    1) AppSettings.openai_api_key (from .env or env)
    2) env var OPENAI_API_KEY
    3) env var openai_api_key
    """
    settings = get_app_settings()

    return (
        settings.openai_api_key
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("openai_api_key")
    )
