from __future__ import annotations

import logging
import logging.config
import os
import sys
from typing import Optional, Dict, Any


# ---------------------------------------------------------------------------
# Internal helper: build a basic logging config
# ---------------------------------------------------------------------------

def _build_logging_config(json_mode: bool = False) -> Dict[str, Any]:
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    service_name = os.getenv("SERVICE_NAME", "erp-ai-service")

    if json_mode:
        # Very simple JSON-style formatter using standard logging
        fmt = (
            '{"level": "%(levelname)s", '
            '"time": "%(asctime)s", '
            '"service": "%(service)s", '
            '"logger": "%(name)s", '
            '"module": "%(module)s", '
            '"func": "%(funcName)s", '
            '"line": %(lineno)d, '
            '"request_id": "%(request_id)s", '
            '"message": %(message)s}'
        )
    else:
        # Human-readable console format
        fmt = (
            "%(asctime)s | %(levelname)-8s | %(service)s | %(name)s | "
            "%(module)s.%(funcName)s:%(lineno)d | [req=%(request_id)s] | %(message)s"
        )

    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": fmt,
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": level,
                "formatter": "default",
                "stream": sys.stdout,
            },
        },
        "root": {
            "level": level,
            "handlers": ["console"],
        },
    }


# ---------------------------------------------------------------------------
# Global flag so we only configure logging once
# ---------------------------------------------------------------------------

_CONFIGURED = False


def _ensure_configured() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    json_mode = os.getenv("LOG_JSON", "false").lower() == "true"

    config = _build_logging_config(json_mode=json_mode)
    logging.config.dictConfig(config)
    _CONFIGURED = True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Return a logger configured with:
      - Level from env LOG_LEVEL (default INFO)
      - Optional JSON format if LOG_JSON=true
      - Extra fields: service, request_id (safe defaults)

    Usage:
        from app.utils.logger import get_logger
        log = get_logger(__name__)
        log.info("Hello", extra={"request_id": "abc-123"})
    """
    _ensure_configured()

    logger_name = name if name is not None else os.getenv("SERVICE_NAME", "erp-ai-service")
    logger = logging.getLogger(logger_name)

    # Attach default extra adapter so missing fields (service, request_id) don't crash formatting
    service_name = os.getenv("SERVICE_NAME", "erp-ai-service")

    class _ContextAdapter(logging.LoggerAdapter):
        def process(self, msg, kwargs):
            extra = kwargs.get("extra", {}) or {}
            # ensure these keys always exist
            extra.setdefault("service", service_name)
            extra.setdefault("request_id", "-")
            kwargs["extra"] = extra
            return msg, kwargs

    return _ContextAdapter(logger, {})


# Convenience module-level logger if you just want `from app.utils.logger import log`
log = get_logger(__name__)
