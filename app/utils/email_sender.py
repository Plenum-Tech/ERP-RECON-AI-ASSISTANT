# app/utils/email_sender.py

from __future__ import annotations

import asyncio
import smtplib
from email.message import EmailMessage
from typing import Any, Dict, Optional

from app.core.config import get_settings

try:
    from app.utils.logger import get_logger

    log = get_logger(__name__)
except Exception:  # pragma: no cover
    import logging

    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(__name__)


def _build_email_message(
    *,
    subject: str,
    to_email: str,
    html_body: str,
    from_email: Optional[str] = None,
) -> EmailMessage:
    settings = get_settings()

    sender = from_email or getattr(settings, "smtp_from_email", None) or "no-reply@example.com"

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = to_email
    msg.set_content("This is an HTML email. Please use an HTML-capable client.")
    msg.add_alternative(html_body, subtype="html")
    return msg


def _send_email_sync(msg: EmailMessage) -> Dict[str, Any]:
    """
    Synchronous SMTP send.

    If SMTP config is missing, we just log and pretend success so that
    local/dev still works without real email infrastructure.
    """
    settings = get_settings()

    host = getattr(settings, "smtp_host", None)
    port = getattr(settings, "smtp_port", None) or 587
    username = getattr(settings, "smtp_username", None)
    password = getattr(settings, "smtp_password", None)
    use_tls = getattr(settings, "smtp_use_tls", True)

    if not host:
        log.warning("SMTP host not configured; simulating email send only.")
        log.info("Simulated email to=%s subject=%s", msg["To"], msg["Subject"])
        return {
            "sent": False,
            "simulated": True,
            "reason": "SMTP host not configured",
            "to": msg["To"],
            "subject": msg["Subject"],
        }

    try:
        if use_tls:
            with smtplib.SMTP(host, port) as server:
                server.starttls()
                if username and password:
                    server.login(username, password)
                server.send_message(msg)
        else:
            with smtplib.SMTP(host, port) as server:
                if username and password:
                    server.login(username, password)
                server.send_message(msg)

        log.info("Email sent to=%s subject=%s", msg["To"], msg["Subject"])
        return {
            "sent": True,
            "simulated": False,
            "to": msg["To"],
            "subject": msg["Subject"],
        }
    except Exception as e:  # pragma: no cover
        log.exception("Error sending email to=%s: %s", msg["To"], e)
        return {
            "sent": False,
            "simulated": False,
            "error": str(e),
            "to": msg["To"],
            "subject": msg["Subject"],
        }


async def send_email(
    *,
    to_email: str,
    subject: str,
    html_body: str,
    from_email: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Async wrapper around the sync SMTP send, so we can call it from FastAPI endpoints.
    """
    msg = _build_email_message(
        subject=subject,
        to_email=to_email,
        html_body=html_body,
        from_email=from_email,
    )

    loop = asyncio.get_running_loop()
    result: Dict[str, Any] = await loop.run_in_executor(None, _send_email_sync, msg)
    return result
