from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict


class DigestItem(BaseModel):
    """
    Lightweight view of a recon item for daily digest / notifications.
    Includes per-item age_days so UI/Bot can show "open for X days".
    """
    model_config = ConfigDict(from_attributes=True)

    id: int
    tx_id: int
    employee_id: Optional[str] = None
    owner_id: Optional[str] = None

    variance_type: Optional[str] = None
    variance_amount: Optional[Decimal] = None
    match_status: str

    opened_at: Optional[datetime] = None
    sla_breached: bool

    # Computed field: how long this item has been open, in days (can be fractional)
    age_days: float


class ReconDigest(BaseModel):
    """
    Daily digest-style summary for recon workspace.

    Intended to drive:
    - Daily email digests
    - Teams cards / notifications
    - Dashboard widgets
    """
    as_of: datetime
    window_days: int

    owner_id: Optional[str] = None

    unmatched_count: int
    avg_age_days: float
    sla_breached_count: int

    # Simple breakdown of unmatched items by variance type
    unmatched_by_variance_type: Dict[str, int]

    # Optional sample items to show as cards/links
    items: List[DigestItem]
