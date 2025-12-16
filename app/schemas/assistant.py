from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import List, Optional, Literal

from pydantic import BaseModel


class AssistantExceptionItem(BaseModel):
    id: int
    tx_id: int
    employee_id: Optional[str]
    variance_type: Optional[str]
    variance_amount: Optional[Decimal]
    match_status: str
    opened_at: Optional[datetime]
    sla_breached: bool
    reason: Optional[str]


class AssistantExceptionSummary(BaseModel):
    employee_id: Optional[str]
    days: int
    unmatched_count: int
    avg_age_days: float
    sla_breached_count: int
    items: List[AssistantExceptionItem]


class FAQEntry(BaseModel):
    question: str
    answer: str


class AssistantQueryRequest(BaseModel):
    """
    Payload for /assistant/query.

    - query: raw user text from Teams / chat.
    - employee_id: optional hint from the bot (if it already knows the employee).
    - days: optional default window for unmatched queries.
    """
    query: str
    employee_id: Optional[str] = None
    days: Optional[int] = None


class AssistantQueryResponse(BaseModel):
    """
    Generic response from /assistant/query.

    kind:
      - 'faq'
      - 'exception_summary'
      - 'item_explanation'
      - 'unknown'
    """
    kind: Literal["faq", "exception_summary", "item_explanation", "unknown"]
    answer_text: str

    faq: Optional[FAQEntry] = None
    exception_summary: Optional[AssistantExceptionSummary] = None
    item: Optional[AssistantExceptionItem] = None
