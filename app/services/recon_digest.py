from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.recon import ReconItem
from app.schemas.digest import ReconDigest, DigestItem


def _ensure_naive_utc(dt: datetime) -> datetime:
    """
    Normalize any datetime to a naive UTC datetime.

    - If tz-aware: convert to UTC, then drop tzinfo.
    - If already naive: return as-is.
    """
    if dt is None:
        return None  # type: ignore[return-value]
    if dt.tzinfo is None:
        return dt
    return dt.astimezone(timezone.utc).replace(tzinfo=None)


async def compute_recon_digest(
    db: AsyncSession,
    *,
    owner_id: Optional[str] = None,
    window_days: int = 1,
) -> ReconDigest:
    """
    Compute a daily-style digest over open recon items.

    - window_days controls how far back we look based on opened_at.
    - owner_id (optional) filters by owner_id, so we can send
      per-owner digests if needed.

    We treat items with match_status == 'unmatched' as open exceptions.
    """
    now = _ensure_naive_utc(datetime.utcnow())
    since = now - timedelta(days=window_days)

    stmt = select(ReconItem).where(ReconItem.match_status == "unmatched")
    stmt = stmt.where(ReconItem.opened_at >= since)

    if owner_id:
        stmt = stmt.where(ReconItem.owner_id == owner_id)

    result = await db.execute(stmt)
    items: List[ReconItem] = result.scalars().all()

    unmatched_count = len(items)

    # Average age in days + per-item age_days
    item_age_days: Dict[int, float] = {}

    if unmatched_count > 0:
        total_age_days = 0.0
        for i in items:
            opened = i.opened_at or now
            opened = _ensure_naive_utc(opened)
            age_delta = now - opened
            age_days = age_delta.total_seconds() / 86400.0
            item_age_days[i.id] = age_days
            total_age_days += age_days

        avg_age_days = total_age_days / unmatched_count
    else:
        avg_age_days = 0.0

    # SLA breach count
    sla_breached_count = sum(1 for i in items if i.sla_breached)

    # Breakdown by variance_type
    unmatched_by_variance_type: Dict[str, int] = {}
    for i in items:
        key = i.variance_type or "UNKNOWN"
        unmatched_by_variance_type[key] = unmatched_by_variance_type.get(key, 0) + 1

    # Map to digest items (thin view, including per-item age_days)
    digest_items: List[DigestItem] = []
    for i in items:
        digest_items.append(
            DigestItem(
                id=i.id,
                tx_id=i.tx_id,
                employee_id=i.employee_id,
                owner_id=i.owner_id,
                variance_type=i.variance_type,
                variance_amount=i.variance_amount,
                match_status=i.match_status,
                opened_at=i.opened_at,
                sla_breached=i.sla_breached,
                age_days=item_age_days.get(i.id, 0.0),
            )
        )

    return ReconDigest(
        as_of=now,
        window_days=window_days,
        owner_id=owner_id,
        unmatched_count=unmatched_count,
        avg_age_days=avg_age_days,
        sla_breached_count=sla_breached_count,
        unmatched_by_variance_type=unmatched_by_variance_type,
        items=digest_items,
    )
