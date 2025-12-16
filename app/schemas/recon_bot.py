from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class ReconBotMessage(BaseModel):
    """
    Incoming message from Teams / UI.

    You can extend this later with org_id, user_id, channel, etc.
    """
    message: str
    org_id: Optional[str] = None
    user_id: Optional[str] = None
    channel: Optional[str] = None


class ReconExceptionItem(BaseModel):
    """
    Thin view of a ReconItem for the bot.

    This is what Teams/Power Automate can use to render a quick list:
    - id + tx_id
    - employee / owner
    - variance / SLA info
    - deep link back into the Recon workspace
    """
    id: int
    tx_id: int
    employee_id: Optional[str] = None
    owner_id: Optional[str] = None
    match_status: str
    variance_type: Optional[str] = None
    variance_amount: Optional[Decimal] = None
    sla_breached: bool = False
    opened_at: Optional[datetime] = None

    # Extra diagnostic bits for UI
    variance_reason: Optional[str] = None

    # Frontend URL to open this item (if configured)
    deep_link: Optional[str] = None


class ReconBotReply(BaseModel):
    """
    Bot reply payload.

    Previously this only had `reply_text`. We are now extending it
    to also return structured exception data for the UI, without
    breaking the existing reply_text behaviour.
    """
    reply_text: str

    # What the LLM intent router decided
    intent: Optional[str] = None  # "exception_lookup" | "faq" | "other"

    # Raw parameters from the intent classifier
    parameters: Dict[str, Any] = {}

    # Top few exceptions, for "exception_lookup" intent
    exceptions: List[ReconExceptionItem] = []
