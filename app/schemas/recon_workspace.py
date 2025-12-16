# app/schemas/recon_workspace.py

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict


class ReconItemOut(BaseModel):
    id: int
    tx_id: int
    source_system: str
    employee_id: Optional[str]
    currency: str
    tx_amount: Decimal
    receipt_amount: Optional[Decimal]
    match_status: str
    variance_type: str
    variance_amount: Optional[Decimal]
    variance_payload: Dict[str, Any]
    suggested_adjustment: Optional[Dict[str, Any]]
    owner_id: Optional[str]

    opened_at: Optional[datetime]
    closed_at: Optional[datetime]
    sla_hours: Optional[int]
    sla_due_at: Optional[datetime]
    sla_breached: bool

    created_at: datetime
    updated_at: datetime

    # Pydantic v2: enable ORM mode
    model_config = ConfigDict(from_attributes=True)


class BankReconItemListResponse(BaseModel):
    total: int
    offset: int
    limit: int
    items: List[ReconItemOut]


class BankReconSummary(BaseModel):
    total_items: int
    by_match_status: Dict[str, int]
    by_variance_type: Dict[str, int]
    matched_count: int
    unmatched_count: int

    total_variance_abs_sum: float
    unmatched_variance_abs_sum: float

    average_open_days: Optional[float]
    sla_breached_count: int


