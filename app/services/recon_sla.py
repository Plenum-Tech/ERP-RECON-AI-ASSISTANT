from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.recon import ReconItem, ReconAction

# Default SLA in hours for recon items (can move to config later)
DEFAULT_SLA_HOURS = 48


def _ensure_naive_utc(dt: datetime) -> datetime:
    """
    Normalize any datetime to a naive UTC datetime.

    - If tz-aware: convert to UTC, then drop tzinfo.
    - If already naive: return as-is.
    """
    if dt.tzinfo is None:
        return dt
    return dt.astimezone(timezone.utc).replace(tzinfo=None)


def apply_default_sla(
    item: ReconItem,
    now: datetime,
    default_sla_hours: int = DEFAULT_SLA_HOURS,
) -> None:
    """
    Ensure an item has SLA fields populated:

    - If sla_hours is None → set to default.
    - If opened_at is None → set to now.
    - If sla_due_at is None and sla_hours is set → opened_at + sla_hours.

    All datetimes are normalized to naive UTC to match TIMESTAMP WITHOUT TIME ZONE.
    """
    # Normalize now to naive UTC
    now = _ensure_naive_utc(now)

    # SLA hours
    if item.sla_hours is None:
        item.sla_hours = default_sla_hours

    # opened_at
    if item.opened_at is None:
        item.opened_at = now
    else:
        item.opened_at = _ensure_naive_utc(item.opened_at)

    # sla_due_at
    if item.sla_due_at is None and item.sla_hours is not None:
        base = item.opened_at or now
        base = _ensure_naive_utc(base)
        item.sla_due_at = base + timedelta(hours=item.sla_hours)
    elif item.sla_due_at is not None:
        # Normalize any pre-existing value
        item.sla_due_at = _ensure_naive_utc(item.sla_due_at)


def update_sla_status_for_item(
    item: ReconItem,
    now: datetime,
) -> bool:
    """
    Update sla_breached for a single item.

    Returns True if any SLA-related field changed.
    """
    # Normalize now
    now = _ensure_naive_utc(now)

    changed = False

    # If item is closed, we don't change sla_breached further (it reflects history)
    if item.closed_at is not None:
        return False

    if item.sla_hours is None or item.sla_due_at is None:
        # nothing to do if SLA is not configured
        return False

    sla_due = _ensure_naive_utc(item.sla_due_at)

    if not item.sla_breached and now > sla_due:
        item.sla_breached = True
        changed = True

    return changed


async def refresh_sla_for_open_items(
    db: AsyncSession,
    default_sla_hours: int = DEFAULT_SLA_HOURS,
    system_actor: str = "sla@system",
) -> Dict[str, Any]:
    """
    Recompute SLA for all open (unmatched) items:

    - Backfill sla_hours / sla_due_at if missing.
    - Normalize datetime fields to naive UTC.
    - Mark sla_breached=True when now > sla_due_at.
    - When an item breaches SLA for the first time, write a ReconAction
      with action_type="sla_breach".

    Intended to be called by a cron/Job or manually via an API endpoint.
    """
    now = _ensure_naive_utc(datetime.utcnow())

    # Only items that are still open (unmatched) and not closed
    stmt = (
        select(ReconItem)
        .where(ReconItem.match_status == "unmatched")
        .where(ReconItem.closed_at.is_(None))
    )
    result = await db.execute(stmt)
    items = result.scalars().all()

    total = len(items)
    updated = 0
    newly_breached = 0

    for item in items:
        before_breached = bool(item.sla_breached)

        # Ensure SLA fields exist and are normalized
        apply_default_sla(item, now, default_sla_hours=default_sla_hours)

        changed = update_sla_status_for_item(item, now)

        if changed:
            updated += 1

            # If SLA just flipped from False -> True, log an audit action
            if not before_breached and item.sla_breached:
                newly_breached += 1

                action = ReconAction(
                    recon_item_id=item.id,
                    action_type="sla_breach",
                    performed_by=system_actor,
                    notes="SLA breached for recon item",
                    payload={
                        "sla_due_at": item.sla_due_at.isoformat()
                        if item.sla_due_at
                        else None,
                        "checked_at": now.isoformat(),
                        "match_status": item.match_status,
                    },
                )
                db.add(action)

            # Touch updated_at whenever SLA-related fields change
            item.updated_at = now

    await db.commit()

    return {
        "checked": total,
        "updated": updated,
        "newly_breached": newly_breached,
        "run_at": now.isoformat() + "Z",
    }
