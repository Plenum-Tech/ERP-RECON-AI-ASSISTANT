from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Literal

from fastapi import (
    APIRouter,
    UploadFile,
    File,
    Depends,
    HTTPException,
    status,
    Query,
)
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.db.session import async_get_db
from app.models.recon import ReconItem
from app.schemas.recon import ReconItemOut
from app.services.bank_reco_store import store_bank_reco_result
from app.services.bank_reco_evaluator import run_matcher
from app.services.recon_workspace import (
    fetch_bank_recon_items,
    summarize_bank_recon,
)

from app.services.recon_resolution import resolve_recon_item, ReconResolutionError, ReconItemNotFound

# Logger
try:
    from app.utils.logger import get_logger

    log = get_logger(__name__)
except Exception:  # pragma: no cover
    import logging

    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(__name__)


router = APIRouter(
    prefix="/bank-reco",
    tags=["Bank Reconciliation"],
)

# ---------------------------------------------------------------------------
# Pydantic response models
# ---------------------------------------------------------------------------


class BankRecoRunSummary(BaseModel):
    statement_lines: int
    ledger_items: int
    matches_1to1: int
    matches_composite_line_from_ledger: int
    matches_composite_ledger_from_lines: int
    matched_total: int
    coverage_statement_pct: float
    coverage_ledger_pct: float
    recommendations_count: int
    recommendations_lines: int

    embeddings_active: bool
    embed_model: Optional[str] = None
    line_embs_shape: Optional[List[int]] = None
    ledger_embs_shape: Optional[List[int]] = None
    matches_with_semantic_signal: int
    recommendation_candidates_with_semantic_sim_ge_0_55: int

    amount_tolerance: Dict[str, float]
    greedy_min_score: float
    weights: Dict[str, float]
    windows: Dict[str, Any]


class BankRecoRunResponse(BaseModel):
    summary: BankRecoRunSummary
    created_items: int
    created_item_ids: List[int]
    dry_run: bool
    store_results: bool
    # Optional lightweight preview so FE can show something without another API call
    normalized_preview: Dict[str, Any]


class ReconWorkspaceSummary(BaseModel):
    total_items: int
    by_match_status: Dict[str, int]
    by_variance_type: Dict[str, int]
    matched_count: int
    unmatched_count: int
    total_variance_abs_sum: float
    unmatched_variance_abs_sum: float
    average_open_days: Optional[float]
    sla_breached_count: int


class ReconWorkspaceResultsResponse(BaseModel):
    items: List[ReconItemOut]
    total: int
    offset: int
    limit: int
    summary: ReconWorkspaceSummary


# ---------------------------------------------------------------------------
# NEW: Resolve (Apply / Ignore) models
# ---------------------------------------------------------------------------

class ResolveReconItemRequest(BaseModel):
    action: Literal["apply", "ignore"]
    note: Optional[str] = None
    # For now, pass actor_id explicitly; later you can read from AAD principal
    actor_id: Optional[str] = None

# ---------------------------------------------------------------------------
# Main endpoint: run + (optionally) store bank reconciliation
# ---------------------------------------------------------------------------


@router.post(
    "/run",
    response_model=BankRecoRunResponse,
    status_code=status.HTTP_200_OK,
    summary="Run bank vs ledger reconciliation and optionally store results",
)
async def run_bank_reco(
    statement_file: UploadFile = File(..., description="Bank statement CSV/XLSX"),
    ledger_file: UploadFile = File(..., description="General ledger CSV/XLSX"),
    tol_abs: float = Query(2.0, description="Absolute tolerance on amount differences"),
    tol_pct: float = Query(0.005, description="Relative tolerance (e.g. 0.005 for 0.5%)"),
    enable_llm_summary: bool = Query(False, description="Generate LLM summary of reconciliation"),
    store_results: bool = Query(True, description="Persist results into erp_ai.recon_items"),
    dry_run: bool = Query(False, description="If True, do not store anything (overrides store_results)"),
    org_id: Optional[str] = Query(None, description="Organisation identifier (optional)"),
    user_id: Optional[str] = Query(None, description="User identifier (optional)"),
    db: AsyncSession = Depends(get_db),
):
    """
    Upload a **bank statement** and a **general ledger** file, run the
    high-precision matcher (`bank_reco_evaluator.run_matcher`), and
    optionally store each statement line as a `ReconItem` in `erp_ai.recon_items`.

    - Uses the same matching logic you shared (1:1 + composite + recommendations).
    - If `dry_run=true` → just returns summary + preview, does not touch the DB.
    - If `store_results=true` & `dry_run=false` → writes one `ReconItem` per statement line.
    """

    # Basic validation on file names
    if not statement_file.filename:
        raise HTTPException(status_code=400, detail="statement_file filename is required")
    if not ledger_file.filename:
        raise HTTPException(status_code=400, detail="ledger_file filename is required")

    allowed_ext = (".csv", ".xlsx", ".xls")
    if not statement_file.filename.lower().endswith(allowed_ext):
        raise HTTPException(
            status_code=400,
            detail=f"statement_file must be one of: {', '.join(allowed_ext)}",
        )
    if not ledger_file.filename.lower().endswith(allowed_ext):
        raise HTTPException(
            status_code=400,
            detail=f"ledger_file must be one of: {', '.join(allowed_ext)}",
        )

    log.info(
        "Starting bank reco run | user_id=%s org_id=%s statement=%s ledger=%s tol_abs=%s tol_pct=%s",
        user_id,
        org_id,
        statement_file.filename,
        ledger_file.filename,
        tol_abs,
        tol_pct,
    )

    # Read uploaded files
    try:
        statement_bytes = await statement_file.read()
        ledger_bytes = await ledger_file.read()
    except Exception as e:
        log.exception("Failed to read uploaded files: %s", e)
        raise HTTPException(
            status_code=500, detail="Failed to read uploaded files"
        ) from e

    # Run the core matcher
    try:
        matcher_result = run_matcher(
            statement_bytes=statement_bytes,
            statement_filename=statement_file.filename,
            ledger_bytes=ledger_bytes,
            ledger_filename=ledger_file.filename,
            tol_abs=tol_abs,
            tol_pct=tol_pct,
            enable_llm_summary=enable_llm_summary,
        )
    except Exception as e:
        log.exception("Bank reconciliation matcher failed: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Bank reconciliation failed: {str(e)}",
        ) from e

    summary_raw: Dict[str, Any] = matcher_result.get("summary") or {}

    # Shape summary into Pydantic model (handles missing keys gracefully)
    try:
        summary = BankRecoRunSummary(**summary_raw)
    except Exception as e:
        # In case summary schema changed, still return raw but log it
        log.warning("Failed to parse BankRecoRunSummary; returning minimal summary. Error: %s", e)
        fallback = {
            "statement_lines": summary_raw.get("statement_lines", 0),
            "ledger_items": summary_raw.get("ledger_items", 0),
            "matches_1to1": summary_raw.get("matches_1to1", 0),
            "matches_composite_line_from_ledger": summary_raw.get(
                "matches_composite_line_from_ledger", 0
            ),
            "matches_composite_ledger_from_lines": summary_raw.get(
                "matches_composite_ledger_from_lines", 0
            ),
            "matched_total": summary_raw.get("matched_total", 0),
            "coverage_statement_pct": summary_raw.get("coverage_statement_pct", 0.0),
            "coverage_ledger_pct": summary_raw.get("coverage_ledger_pct", 0.0),
            "recommendations_count": summary_raw.get("recommendations_count", 0),
            "recommendations_lines": summary_raw.get("recommendations_lines", 0),
            "embeddings_active": summary_raw.get("embeddings_active", False),
            "embed_model": summary_raw.get("embed_model"),
            "line_embs_shape": summary_raw.get("line_embs_shape"),
            "ledger_embs_shape": summary_raw.get("ledger_embs_shape"),
            "matches_with_semantic_signal": summary_raw.get(
                "matches_with_semantic_signal", 0
            ),
            "recommendation_candidates_with_semantic_sim_ge_0_55": summary_raw.get(
                "recommendation_candidates_with_semantic_sim_ge_0_55", 0
            ),
            "amount_tolerance": summary_raw.get("amount_tolerance", {}),
            "greedy_min_score": summary_raw.get("greedy_min_score", 0.0),
            "weights": summary_raw.get("weights", {}),
            "windows": summary_raw.get("windows", {}),
        }
        summary = BankRecoRunSummary(**fallback)

    created_items: List[ReconItem] = []
    created_ids: List[int] = []

    effective_store = store_results and not dry_run

    if effective_store:
        try:
            # If you have UC1 tx rows already imported, map statement row_id -> tx_id here
            tx_id_map: Optional[Dict[int, int]] = None  # or build a real mapping

            # NOTE: store_bank_reco_result now manages its own flush/commit/rollback.
            created_items = await store_bank_reco_result(
                db=db,
                matcher_result=matcher_result,
                org_id=org_id,
                user_id=user_id,
                default_currency="AED",
                source_system="BANK",
                tx_id_map=tx_id_map,
                run_meta={
                    "statement_filename": statement_file.filename,
                    "ledger_filename": ledger_file.filename,
                    "tol_abs": tol_abs,
                    "tol_pct": tol_pct,
                },
            )

            created_ids = [int(item.id) for item in created_items if item.id is not None]

            log.info(
                "Bank reco run stored %s ReconItem rows | user_id=%s org_id=%s",
                len(created_items),
                user_id,
                org_id,
            )
        except Exception as e:
            # store_bank_reco_result already rolls back on SQLAlchemyError
            log.exception("Failed to store bank reconciliation results: %s", e)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to store bank reconciliation results: {str(e)}",
            ) from e
    else:
        log.info(
            "Bank reco run in dry-run mode; results not stored | user_id=%s org_id=%s",
            user_id,
            org_id,
        )

    # Lightweight preview only (not full massive payload)
    normalized_preview = matcher_result.get("normalized_preview") or {
        "statement_sample": [],
        "ledger_sample": [],
    }

    return BankRecoRunResponse(
        summary=summary,
        created_items=len(created_items),
        created_item_ids=created_ids,
        dry_run=dry_run,
        store_results=effective_store,
        normalized_preview=normalized_preview,
    )


# ---------------------------------------------------------------------------
# Workspace results endpoint (filtered + paginated + summary)
# ---------------------------------------------------------------------------


@router.get(
    "/results",
    response_model=ReconWorkspaceResultsResponse,
    summary="List stored BANK recon items with filters + summary",
)
async def list_bank_reco_results(
    match_status: Optional[str] = Query(
        None,
        description=(
            "Filter by match_status. Supports comma-separated values, e.g.: "
            "'matched,unmatched'"
        ),
    ),
    variance_type: Optional[str] = Query(
        None,
        description=(
            "Filter by variance_type (e.g., bank_gl_1to1, bank_gl_unmatched). "
            "Supports comma-separated list."
        ),
    ),
    min_tx_amount: Optional[float] = Query(
        None,
        description="Minimum transaction amount (statement signed_amount)",
    ),
    max_tx_amount: Optional[float] = Query(
        None,
        description="Maximum transaction amount (statement signed_amount)",
    ),
    min_variance: Optional[float] = Query(
        None,
        description="Minimum variance_amount (absolute, in currency units)",
    ),
    max_variance: Optional[float] = Query(
        None,
        description="Maximum variance_amount (absolute, in currency units)",
    ),
    owner_id: Optional[str] = Query(
        None,
        description="Filter by owner_id (e.g., analyst / ops owner)",
    ),
    employee_id: Optional[str] = Query(
        None,
        description="Filter by employee_id (for salary / payroll cases)",
    ),
    created_from: Optional[datetime] = Query(
        None,
        description="Filter items created_at >= this datetime (ISO 8601)",
    ),
    created_to: Optional[datetime] = Query(
        None,
        description="Filter items created_at <= this datetime (ISO 8601)",
    ),
    offset: int = Query(
        0,
        ge=0,
        description="Pagination offset (0-based)",
    ),
    limit: int = Query(
        50,
        ge=1,
        le=500,
        description="Page size (max 500)",
    ),
    db: AsyncSession = Depends(async_get_db),
):
    """
    List stored bank reconciliation items from `erp_ai.recon_items` (source_system='BANK'),
    with the same filters your workspace UI will use, and return a summary block
    that matches the visible grid.

    Filters:
    - `match_status`: matched / unmatched, supports comma-separated list.
    - `variance_type`: e.g. `bank_gl_1to1`, `bank_gl_unmatched`, supports comma-separated.
    - `min_tx_amount`, `max_tx_amount`: filter on statement-side transaction size.
    - `min_variance`, `max_variance`: filter on variance_amount (absolute).
    - `owner_id`, `employee_id`: user/employee scoping for ops.
    - `created_from`, `created_to`: created_at date range.
    """

    try:
        items_orm, total = await fetch_bank_recon_items(
            db=db,
            source_system="BANK",
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
    except Exception as e:
        log.exception("Failed to fetch bank recon items: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch bank recon items: {str(e)}",
        ) from e

    items_out = [ReconItemOut.from_orm(i) for i in items_orm]

    # Summary uses same filters, but without amount/variance bounds (by design)
    try:
        summary_raw = await summarize_bank_recon(
            db=db,
            source_system="BANK",
            match_status=match_status,
            variance_type=variance_type,
            owner_id=owner_id,
            employee_id=employee_id,
            created_from=created_from,
            created_to=created_to,
        )
        summary = ReconWorkspaceSummary(**summary_raw)
    except Exception as e:
        log.exception("Failed to summarize bank recon workspace: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to summarize bank recon workspace: {str(e)}",
        ) from e

    return ReconWorkspaceResultsResponse(
        items=items_out,
        total=total,
        offset=offset,
        limit=limit,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# NEW: Apply / Ignore with audit
# ---------------------------------------------------------------------------

@router.post(
    "/items/{item_id}/resolve",
    response_model=ReconItemOut,
    status_code=status.HTTP_200_OK,
    summary="Resolve a recon item as applied or ignored with audit + SLA update",
)
async def resolve_bank_reco_item(
    item_id: int,
    payload: ResolveReconItemRequest,
    db: AsyncSession = Depends(async_get_db),
):
    """
    Apply / Ignore a recon item.

    - action: "apply" or "ignore"
    - note: optional free-text note
    - actor_id: optional user identifier (later can be taken from AAD)

    Updates:
    - resolution_status, resolution_note, resolved_by, resolved_at
    - closed_at, sla_breached (if sla_due_at is set)
    """
    try:
        item = await resolve_recon_item(
            db,
            item_id=item_id,
            action=payload.action,
            note=payload.note,
            actor_id=payload.actor_id,
        )
    except ReconItemNotFound:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"ReconItem {item_id} not found",
        )
    except ReconResolutionError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    return ReconItemOut.from_orm(item)
