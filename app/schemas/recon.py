from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Optional, Dict, Any

from pydantic import BaseModel, ConfigDict, Field


class ReconItemOut(BaseModel):
    # Pydantic v2: replaces `class Config: orm_mode = True`
    model_config = ConfigDict(from_attributes=True)

    id: int
    tx_id: int
    source_system: Optional[str]

    employee_id: Optional[str]
    currency: str
    tx_amount: Decimal
    receipt_amount: Optional[Decimal]

    match_status: str
    variance_type: Optional[str]
    variance_amount: Optional[Decimal]
    variance_payload: Optional[Dict[str, Any]]

    suggested_adjustment: Optional[Dict[str, Any]]

    owner_id: Optional[str]
    opened_at: datetime
    closed_at: Optional[datetime]

    sla_hours: Optional[int]
    sla_due_at: Optional[datetime]
    sla_breached: bool

    created_at: datetime
    updated_at: datetime


class ApplyReconResponse(BaseModel):
    id: int
    match_status: str
    message: str

class ReconItemOut(BaseModel):
    id: int
    tx_id: int
    source_system: str
    employee_id: Optional[str] = None
    currency: str
    tx_amount: float
    receipt_amount: Optional[float] = None
    match_status: str
    variance_type: str
    variance_amount: Optional[float] = None
    variance_payload: dict = Field(default_factory=dict)
    suggested_adjustment: Optional[dict] = None
    owner_id: Optional[str] = None
    opened_at: Optional[datetime] = None
    closed_at: Optional[datetime] = None
    sla_hours: Optional[int] = None
    sla_due_at: Optional[datetime] = None
    sla_breached: Optional[bool] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True  # SQLAlchemy compatibility
        