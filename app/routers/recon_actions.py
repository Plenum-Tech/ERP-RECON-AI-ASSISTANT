from typing import Any, Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.schemas.recon_actions import (
    ReconCaseApplyRequest,
    ReconCaseIgnoreRequest,
    ReconCaseActionRead,
)
from app.services import recon_actions as service


# -------------------------------------------------------------------
# TEMP: simple get_current_user stub so router works without real auth
# -------------------------------------------------------------------
async def get_current_user() -> Any:
    """
    Dummy current user for local/dev when auth is not wired.

    Replace this with a real dependency once you plug in AAD / JWT.
    """

    class _User:
        email = "system@local"
        id = "system"

    return _User()


router = APIRouter(prefix="/recon", tags=["reconciliation-actions"])


@router.post(
    "/cases/{case_id}/apply",
    response_model=ReconCaseActionRead,
)
async def apply_recon_case(
    case_id: int,
    body: ReconCaseApplyRequest,
    db: AsyncSession = Depends(get_db),
    current_user: Any = Depends(get_current_user),
):
    """
    Apply the suggested adjustment for a reconciliation case.

    - Respects `suggested_adjustment.auto_apply_safe` unless `force=true`.
    - Marks case as matched + closed.
    - Writes an audit action with posting stub.
    """
    performed_by = (
        getattr(current_user, "email", None)
        or getattr(current_user, "id", None)
        or "unknown"
    )

    action = await service.apply_case(
        db=db,
        case_id=case_id,
        performed_by=performed_by,
        notes=body.notes,
        force=body.force,
    )
    return action


@router.post(
    "/cases/{case_id}/ignore",
    response_model=ReconCaseActionRead,
)
async def ignore_recon_case(
    case_id: int,
    body: ReconCaseIgnoreRequest,
    db: AsyncSession = Depends(get_db),
    current_user: Any = Depends(get_current_user),
):
    """
    Ignore a reconciliation case.

    - Sets match_status to 'ignored'.
    - Closes the case.
    - Writes an audit action with reason (if provided).
    """
    performed_by = (
        getattr(current_user, "email", None)
        or getattr(current_user, "id", None)
        or "unknown"
    )

    action = await service.ignore_case(
        db=db,
        case_id=case_id,
        performed_by=performed_by,
        notes=body.notes,
        reason=body.reason,
    )
    return action


@router.post("/cases/auto-apply-safe")
async def auto_apply_safe_recon_cases(
    limit: Optional[int] = Query(
        default=None,
        ge=1,
        description="Optional max number of cases to process in this call",
    ),
    db: AsyncSession = Depends(get_db),
    current_user: Any = Depends(get_current_user),
):
    """
    Bulk-apply all unmatched cases whose suggested_adjustment.auto_apply_safe == true.

    This is intended for "safe auto-reconcile within tolerance" operations.

    Returns a summary:
    {
      "applied_count": ...,
      "applied_ids": [...],
      "skipped_count": ...,
      "skipped": [{"id": ..., "reason": "..."}]
    }
    """
    performed_by = (
        getattr(current_user, "email", None)
        or getattr(current_user, "id", None)
        or "unknown"
    )

    result = await service.auto_apply_safe_cases(
        db=db,
        performed_by=performed_by,
        limit=limit,
    )
    return result
