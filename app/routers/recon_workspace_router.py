# app/routers/recon_workspace_router.py

from __future__ import annotations

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.schemas.recon_workspace import (
    BankReconItemListResponse,
    BankReconSummary,
    ReconItemOut,
)
from app.services.recon_workspace import (
    fetch_bank_recon_items,
    summarize_bank_recon,
)

router = APIRouter(
    prefix="/recon/bank",
    tags=["Bank Reconciliation Workspace"],
)


@router.get("/items", response_model=BankReconItemListResponse)
async def list_bank_recon_items(
    source_system: str = Query("BANK", description="Source system tag, default BANK."),
    match_status: Optional[str] = Query(
        None,
        description='Filter by match_status. Example: "matched" or "matched,unmatched".',
    ),
    variance_type: Optional[str] = Query(
        None,
        description='Filter by variance_type. Example: "bank_gl_1to1,bank_gl_unmatched".',
    ),
    min_tx_amount: Optional[float] = Query(None),
    max_tx_amount: Optional[float] = Query(None),
    min_variance: Optional[float] = Query(None),
    max_variance: Optional[float] = Query(None),
    owner_id: Optional[str] = Query(None),
    employee_id: Optional[str] = Query(None),
    created_from: Optional[datetime] = Query(
        None,
        description="Filter items created on/after this UTC datetime.",
    ),
    created_to: Optional[datetime] = Query(
        None,
        description="Filter items created on/before this UTC datetime.",
    ),
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=500),
    db: AsyncSession = Depends(get_db),
) -> BankReconItemListResponse:
    """
    List bank reconciliation items (workspace view) with filters + pagination.
    """

    items, total = await fetch_bank_recon_items(
        db,
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
        offset=offset,
        limit=limit,
    )

    return BankReconItemListResponse(
        total=total,
        offset=offset,
        limit=limit,
        items=items,
    )


@router.get("/summary", response_model=BankReconSummary)
async def get_bank_recon_summary(
    source_system: str = Query("BANK", description="Source system tag, default BANK."),
    match_status: Optional[str] = Query(
        None,
        description='Filter by match_status (same semantics as /items).',
    ),
    variance_type: Optional[str] = Query(
        None,
        description='Filter by variance_type (same semantics as /items).',
    ),
    owner_id: Optional[str] = Query(None),
    employee_id: Optional[str] = Query(None),
    created_from: Optional[datetime] = Query(
        None,
        description="Filter items created on/after this UTC datetime.",
    ),
    created_to: Optional[datetime] = Query(
        None,
        description="Filter items created on/before this UTC datetime.",
    ),
    db: AsyncSession = Depends(get_db),
) -> BankReconSummary:
    """
    Summary KPIs for the bank reconciliation workspace.

    Uses the same filters as /items so that the numbers match the list view.
    """

    summary_dict = await summarize_bank_recon(
        db,
        source_system=source_system,
        match_status=match_status,
        variance_type=variance_type,
        owner_id=owner_id,
        employee_id=employee_id,
        created_from=created_from,
        created_to=created_to,
    )

    return BankReconSummary(**summary_dict)
