# app/services/recon_workspace.py

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.recon import ReconItem


def _build_filters(
    *,
    source_system: str,
    match_status: Optional[str] = None,
    variance_type: Optional[str] = None,
    min_tx_amount: Optional[float] = None,
    max_tx_amount: Optional[float] = None,
    min_variance: Optional[float] = None,
    max_variance: Optional[float] = None,
    owner_id: Optional[str] = None,
    employee_id: Optional[str] = None,
    created_from: Optional[datetime] = None,
    created_to: Optional[datetime] = None,
) -> List[Any]:
    """Build SQLAlchemy filter conditions for ReconItem."""
    conds: List[Any] = [ReconItem.source_system == source_system]

    if match_status:
        # allow comma-separated list: "matched,unmatched"
        if "," in match_status:
            statuses = [s.strip() for s in match_status.split(",") if s.strip()]
            if statuses:
                conds.append(ReconItem.match_status.in_(statuses))
        else:
            conds.append(ReconItem.match_status == match_status)

    if variance_type:
        if "," in variance_type:
            vts = [v.strip() for v in variance_type.split(",") if v.strip()]
            if vts:
                conds.append(ReconItem.variance_type.in_(vts))
        else:
            conds.append(ReconItem.variance_type == variance_type)

    if min_tx_amount is not None:
        conds.append(ReconItem.tx_amount >= min_tx_amount)
    if max_tx_amount is not None:
        conds.append(ReconItem.tx_amount <= max_tx_amount)

    if min_variance is not None:
        conds.append(ReconItem.variance_amount >= min_variance)
    if max_variance is not None:
        conds.append(ReconItem.variance_amount <= max_variance)

    if owner_id:
        conds.append(ReconItem.owner_id == owner_id)

    if employee_id:
        conds.append(ReconItem.employee_id == employee_id)

    if created_from is not None:
        conds.append(ReconItem.created_at >= created_from)
    if created_to is not None:
        conds.append(ReconItem.created_at <= created_to)

    return conds


async def fetch_bank_recon_items(
    db: AsyncSession,
    *,
    source_system: str = "BANK",
    match_status: Optional[str] = None,
    variance_type: Optional[str] = None,
    min_tx_amount: Optional[float] = None,
    max_tx_amount: Optional[float] = None,
    min_variance: Optional[float] = None,
    max_variance: Optional[float] = None,
    owner_id: Optional[str] = None,
    employee_id: Optional[str] = None,
    created_from: Optional[datetime] = None,
    created_to: Optional[datetime] = None,
    offset: int = 0,
    limit: int = 50,
) -> Tuple[List[ReconItem], int]:
    """Return filtered list of ReconItems and total count for pagination."""

    conds = _build_filters(
        source_system=source_system,
        match_status=match_status,
        variance_type=variance_type,
        min_tx_amount=min_tx_amount,
        max_tx_amount=max_tx_amount,
        min_variance=min_variance,
        max_variance=max_variance,
        owner_id=owner_id,
        employee_id=employee_id,
        created_from=created_from,
        created_to=created_to,
    )

    where_clause = and_(*conds) if conds else True

    # main query
    stmt = (
        select(ReconItem)
        .where(where_clause)
        .order_by(ReconItem.created_at.desc())
        .offset(offset)
        .limit(limit)
    )
    result = await db.execute(stmt)
    items = result.scalars().all()

    # total count
    count_stmt = select(func.count()).select_from(ReconItem).where(where_clause)
    total = (await db.execute(count_stmt)).scalar_one()

    return items, total


async def summarize_bank_recon(
    db: AsyncSession,
    *,
    source_system: str = "BANK",
    match_status: Optional[str] = None,
    variance_type: Optional[str] = None,
    owner_id: Optional[str] = None,
    employee_id: Optional[str] = None,
    created_from: Optional[datetime] = None,
    created_to: Optional[datetime] = None,
) -> Dict[str, Any]:
    """
    Compute high-level summary stats for the workspace, using the same filters
    as the list view (so the UI summary matches what the user is seeing).
    """
    conds = _build_filters(
        source_system=source_system,
        match_status=match_status,
        variance_type=variance_type,
        min_tx_amount=None,
        max_tx_amount=None,
        min_variance=None,
        max_variance=None,
        owner_id=owner_id,
        employee_id=employee_id,
        created_from=created_from,
        created_to=created_to,
    )

    where_clause = and_(*conds) if conds else True

    # total items
    total_items_stmt = select(func.count()).select_from(ReconItem).where(where_clause)
    total_items = (await db.execute(total_items_stmt)).scalar_one()

    # by match_status
    by_match_status_stmt = (
        select(ReconItem.match_status, func.count())
        .where(where_clause)
        .group_by(ReconItem.match_status)
    )
    rows = (await db.execute(by_match_status_stmt)).all()
    by_match_status: Dict[str, int] = {row[0]: row[1] for row in rows}

    matched_count = by_match_status.get("matched", 0)
    unmatched_count = by_match_status.get("unmatched", 0)

    # by variance_type
    by_variance_type_stmt = (
        select(ReconItem.variance_type, func.count())
        .where(where_clause)
        .group_by(ReconItem.variance_type)
    )
    rows = (await db.execute(by_variance_type_stmt)).all()
    by_variance_type: Dict[str, int] = {row[0]: row[1] for row in rows if row[0] is not None}

    # total variance abs sum (all items)
    total_variance_abs_stmt = select(
        func.coalesce(func.sum(func.abs(ReconItem.variance_amount)), 0.0)
    ).where(where_clause)
    total_variance_abs_sum = float((await db.execute(total_variance_abs_stmt)).scalar_one())

    # unmatched variance abs sum
    unmatched_variance_abs_stmt = select(
        func.coalesce(func.sum(func.abs(ReconItem.variance_amount)), 0.0)
    ).where(where_clause, ReconItem.match_status == "unmatched")
    unmatched_variance_abs_sum = float(
        (await db.execute(unmatched_variance_abs_stmt)).scalar_one()
    )

    # average open days (now - opened_at; closed_at if present)
    # NOTE: this is DB-side; opened_at must be non-null
    age_days_expr = (
        func.extract(
            "epoch",
            func.coalesce(ReconItem.closed_at, func.now()) - ReconItem.opened_at,
        )
        / 86400.0
    )
    avg_open_stmt = select(func.avg(age_days_expr)).where(
        where_clause, ReconItem.opened_at.isnot(None)
    )
    avg_open_val = (await db.execute(avg_open_stmt)).scalar_one()
    average_open_days = float(avg_open_val) if avg_open_val is not None else None

    # SLA breached count
    sla_breached_stmt = select(func.count()).select_from(ReconItem).where(
        where_clause, ReconItem.sla_breached.is_(True)
    )
    sla_breached_count = (await db.execute(sla_breached_stmt)).scalar_one()

    return {
        "total_items": total_items,
        "by_match_status": by_match_status,
        "by_variance_type": by_variance_type,
        "matched_count": matched_count,
        "unmatched_count": unmatched_count,
        "total_variance_abs_sum": total_variance_abs_sum,
        "unmatched_variance_abs_sum": unmatched_variance_abs_sum,
        "average_open_days": average_open_days,
        "sla_breached_count": sla_breached_count,
    }
