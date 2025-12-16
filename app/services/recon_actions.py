from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.recon import ReconItem, ReconAction  # ✅ updated imports


async def _get_item_for_update(db: AsyncSession, item_id: int) -> ReconItem:
    """
    Load a ReconItem row with row-level lock for update.
    """
    stmt = (
        select(ReconItem)
        .where(ReconItem.id == item_id)
        .with_for_update()
    )
    result = await db.execute(stmt)
    item = result.scalar_one_or_none()
    if item is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Reconciliation item not found",
        )
    return item


async def apply_case(
    db: AsyncSession,
    case_id: int,
    performed_by: str,
    notes: Optional[str] = None,
    force: bool = False,
) -> ReconAction:
    """
    Apply the suggested adjustment for a ReconItem.

    - Validates that a suggested_adjustment exists
    - Enforces auto_apply_safe unless force=True
    - Marks the item as matched/closed
    - Writes a ReconAction audit row
    """
    # We keep the external param name "case_id" for backward-compat with routers,
    # but internally we now work with ReconItem.
    item = await _get_item_for_update(db, case_id)

    # Guardrails
    if item.match_status == "matched":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Case is already matched",
        )

    suggested = item.suggested_adjustment or {}
    auto_safe = bool(suggested.get("auto_apply_safe"))
    amount = suggested.get("amount")

    if amount is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No suggested adjustment available for this case",
        )

    if not auto_safe and not force:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Suggested adjustment not marked as auto-apply safe. "
                "Use force=true to override."
            ),
        )

    # ---- Stub: here you’d call ERP / queue a job to post the journal ----
    posting_instruction: Dict[str, Any] = {
        "status": "queued",  # or "posted" if synchronous
        "engine": "bank_reco_v1",
        "recon_item_id": item.id,
        "amount": amount,
        "currency": suggested.get("currency"),
        "reason": suggested.get("reason"),
        "lines": suggested.get("lines", []),
    }
    # ----------------------------------------------------------------------#

    previous_status = item.match_status
    previous_closed_at = item.closed_at

    now = datetime.utcnow()

    # Update item state
    item.match_status = "matched"  # or "applied" if you later want a separate flag
    item.closed_at = now
    item.updated_at = now

    # Also update resolution summary fields (for quick reporting)
    item.resolution_status = "applied"
    item.resolution_note = notes
    item.resolved_by = performed_by
    item.resolved_at = now

    # Create audit action
    action = ReconAction(
        recon_item_id=item.id,  # ✅ correct FK field
        action_type="apply",
        performed_by=performed_by,
        notes=notes,
        payload={
            "previous_status": previous_status,
            "previous_closed_at": (
                previous_closed_at.isoformat() if previous_closed_at else None
            ),
            "suggested_adjustment": suggested,
            "posting_instruction": posting_instruction,
            "force": force,
        },
        adjustment_snapshot=suggested or None,  # freeze at time of apply
    )

    db.add(action)
    await db.commit()
    await db.refresh(action)

    return action


async def ignore_case(
    db: AsyncSession,
    case_id: int,
    performed_by: str,
    notes: Optional[str] = None,
    reason: Optional[str] = None,
) -> ReconAction:
    """
    Mark a ReconItem as ignored and close it, with audit trail.
    """
    item = await _get_item_for_update(db, case_id)

    if item.match_status == "ignored":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Case is already ignored",
        )

    previous_status = item.match_status
    previous_closed_at = item.closed_at

    now = datetime.utcnow()

    # Mark as ignored + close
    item.match_status = "ignored"
    item.closed_at = now
    item.updated_at = now

    # Update resolution summary fields
    item.resolution_status = "ignored"
    item.resolution_note = notes or reason
    item.resolved_by = performed_by
    item.resolved_at = now

    payload: Dict[str, Any] = {
        "previous_status": previous_status,
        "previous_closed_at": (
            previous_closed_at.isoformat() if previous_closed_at else None
        ),
    }
    if reason:
        payload["ignore_reason"] = reason

    action = ReconAction(
        recon_item_id=item.id,  # ✅ correct FK field
        action_type="ignore",
        performed_by=performed_by,
        notes=notes,
        payload=payload,
    )

    db.add(action)
    await db.commit()
    await db.refresh(action)

    return action


async def auto_apply_safe_cases(
    db: AsyncSession,
    performed_by: str,
    limit: Optional[int] = None,
    note: str = "Auto-applied by system (auto_apply_safe=true)",
) -> Dict[str, Any]:
    """
    Scan unmatched ReconItems and automatically apply all that are marked
    auto_apply_safe=true in suggested_adjustment.

    Returns a summary dict with applied & skipped IDs.
    """
    stmt = select(ReconItem).where(ReconItem.match_status == "unmatched")
    result = await db.execute(stmt)
    items: List[ReconItem] = result.scalars().all()

    if limit is not None and limit > 0:
        items = items[:limit]

    applied_ids: List[int] = []
    skipped: List[Dict[str, Any]] = []

    for item in items:
        suggested = item.suggested_adjustment or {}
        if not suggested.get("auto_apply_safe"):
            skipped.append(
                {"id": item.id, "reason": "auto_apply_safe_false_or_missing"}
            )
            continue

        try:
            # Reuse the same logic & guardrails as manual apply.
            await apply_case(
                db=db,
                case_id=item.id,
                performed_by=performed_by,
                notes=note,
                force=False,  # rely on auto_apply_safe instead of forcing
            )
            applied_ids.append(item.id)
        except HTTPException as exc:
            # If for some reason it fails (e.g. amount missing), record a skip.
            skipped.append({"id": item.id, "reason": exc.detail})

    return {
        "applied_count": len(applied_ids),
        "applied_ids": applied_ids,
        "skipped_count": len(skipped),
        "skipped": skipped,
    }
