from datetime import datetime
from typing import Optional, Any, Dict

from pydantic import BaseModel


class ReconCaseActionBase(BaseModel):
    notes: Optional[str] = None


class ReconCaseApplyRequest(ReconCaseActionBase):
    # if False → must respect suggested_adjustment.auto_apply_safe
    # if True  → allow manual override (still logged)
    force: bool = False


class ReconCaseIgnoreRequest(ReconCaseActionBase):
    # optional more specific reason to show in UI / reports
    reason: Optional[str] = None


class ReconCaseActionRead(BaseModel):
    id: int
    case_id: int
    action_type: str
    performed_by: str
    performed_at: datetime
    notes: Optional[str]

    class Config:
        from_attributes = True  # Pydantic v2; use orm_mode = True for v1
