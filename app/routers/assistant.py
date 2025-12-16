from __future__ import annotations

from datetime import datetime, timedelta
import re
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.models.recon import ReconItem
from app.schemas.assistant import (
    AssistantExceptionSummary,
    AssistantExceptionItem,
    FAQEntry,
    AssistantQueryRequest,
    AssistantQueryResponse,
)

# from app.api.deps import get_current_user  # plug in AAD auth later

router = APIRouter(prefix="/assistant", tags=["assistant"])


# ---------------------------------------------------------------------------
# In-memory FAQ list (placeholder for "top 30 finance FAQs")
# ---------------------------------------------------------------------------

_FAQS: List[FAQEntry] = [
    FAQEntry(
        question="What is a reconciliation variance?",
        answer=(
            "A reconciliation variance is the difference between two related records "
            "(for example, card transaction vs. receipt vs. ERP entry). "
            "Small differences may be due to FX, tips, or partial payments."
        ),
    ),
    FAQEntry(
        question="What is FX variance?",
        answer=(
            "FX variance occurs when a transaction in a foreign currency settles "
            "at a different exchange rate than the rate used on the receipt or ERP entry, "
            "leading to a small gain or loss in home currency."
        ),
    ),
    FAQEntry(
        question="What does 'Apply adjustment' do?",
        answer=(
            "'Apply adjustment' posts the system-suggested journal entry to clear the variance "
            "(for example, debit FX loss and credit card clearing), and marks the recon item as adjusted "
            "with a full audit trail."
        ),
    ),
    # Add more entries over time until you reach your 'top 30' set
]


@router.get("/faqs", response_model=List[FAQEntry])
async def list_faqs():
    """
    Lite FAQ endpoint for the Teams bot: returns a list of common finance FAQs.
    In production, this could be replaced by a knowledge base or RAG-backed assistant.
    """
    return _FAQS


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

async def _fetch_unmatched_summary(
    db: AsyncSession,
    employee_id: Optional[str],
    days: int,
) -> AssistantExceptionSummary:
    """
    Shared logic for:
    - /assistant/exception-status
    - /assistant/query (exception_summary mode)
    """
    if days <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="days must be > 0",
        )

    now = datetime.utcnow()
    since = now - timedelta(days=days)

    stmt = select(ReconItem).where(ReconItem.match_status == "unmatched")
    stmt = stmt.where(ReconItem.opened_at >= since)

    if employee_id:
        stmt = stmt.where(ReconItem.employee_id == employee_id)

    result = await db.execute(stmt)
    items = result.scalars().all()

    unmatched_count = len(items)
    sla_breached_count = sum(1 for i in items if i.sla_breached)

    # Average age in days (0 if none)
    if unmatched_count > 0:
        total_age_days = 0.0
        for i in items:
            opened = i.opened_at or now
            age = now - opened
            total_age_days += age.total_seconds() / 86400.0
        avg_age_days = total_age_days / unmatched_count
    else:
        avg_age_days = 0.0

    assistant_items: List[AssistantExceptionItem] = []
    for i in items:
        reason: Optional[str] = None
        if isinstance(i.variance_payload, dict):
            # try to pull a human-friendly explanation
            reason = (
                i.variance_payload.get("explanation")
                or i.variance_payload.get("variance_type")
            )

        assistant_items.append(
            AssistantExceptionItem(
                id=i.id,
                tx_id=i.tx_id,
                employee_id=i.employee_id,
                variance_type=i.variance_type,
                variance_amount=i.variance_amount,
                match_status=i.match_status,
                opened_at=i.opened_at,
                sla_breached=i.sla_breached,
                reason=reason,
            )
        )

    return AssistantExceptionSummary(
        employee_id=employee_id,
        days=days,
        unmatched_count=unmatched_count,
        avg_age_days=avg_age_days,
        sla_breached_count=sla_breached_count,
        items=assistant_items,
    )


async def _fetch_single_item(
    db: AsyncSession,
    item_id: int,
) -> AssistantExceptionItem:
    result = await db.execute(
        select(ReconItem).where(ReconItem.id == item_id)
    )
    i = result.scalar_one_or_none()

    if not i:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"ReconItem {item_id} not found",
        )

    reason: Optional[str] = None
    if isinstance(i.variance_payload, dict):
        reason = (
            i.variance_payload.get("explanation")
            or i.variance_payload.get("variance_type")
        )

    return AssistantExceptionItem(
        id=i.id,
        tx_id=i.tx_id,
        employee_id=i.employee_id,
        variance_type=i.variance_type,
        variance_amount=i.variance_amount,
        match_status=i.match_status,
        opened_at=i.opened_at,
        sla_breached=i.sla_breached,
        reason=reason,
    )


def _match_best_faq(query: str) -> Optional[FAQEntry]:
    """
    Very naive FAQ matcher:
    - lowercases query
    - returns first FAQ whose question text appears in the query,
      or whose key phrase appears in the query.
    """
    q = query.lower()

    # Simple keyword hints
    if "fx variance" in q or "foreign exchange" in q:
        for faq in _FAQS:
            if "fx variance" in faq.question.lower():
                return faq

    if "reconciliation variance" in q or "recon variance" in q:
        for faq in _FAQS:
            if "reconciliation variance" in faq.question.lower():
                return faq

    if "apply adjustment" in q or "apply the adjustment" in q:
        for faq in _FAQS:
            if "apply adjustment" in faq.question.lower():
                return faq

    # Fallback: very rough substring match on question text
    for faq in _FAQS:
        if any(word in q for word in faq.question.lower().split()):
            return faq

    return None


def _extract_days_from_text(query: str) -> Optional[int]:
    """
    Look for patterns like 'last 7 days' or 'past 30 days'.
    """
    q = query.lower()
    m = re.search(r"(last|past)\s+(\d+)\s+day", q)
    if m:
        try:
            return int(m.group(2))
        except ValueError:
            return None
    return None


def _extract_item_id_from_text(query: str) -> Optional[int]:
    """
    Look for 'item 123' or 'recon item 123' or 'id 123'.
    """
    q = query.lower()

    for pattern in [
        r"item\s+(\d+)",
        r"id\s+(\d+)",
        r"recon\s+item\s+(\d+)",
    ]:
        m = re.search(pattern, q)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                continue
    return None


def _extract_employee_from_text(query: str) -> Optional[str]:
    """
    Very naive heuristic:
    - look for 'for <word>' and return that word, stripping punctuation.
    """
    q = query.lower()
    m = re.search(r"for\s+([a-z0-9_\-]+)", q)
    if m:
        return m.group(1)
    return None


# ---------------------------------------------------------------------------
# Exception status by employee/date (explicit endpoint)
# ---------------------------------------------------------------------------

@router.get(
    "/exception-status",
    response_model=AssistantExceptionSummary,
)
async def exception_status_by_employee(
    employee_id: Optional[str] = None,
    days: int = 7,
    db: AsyncSession = Depends(get_db),
    # current_user=Depends(get_current_user),
):
    """
    Explicit endpoint (used by bot or other services) to get exception status.
    """
    return await _fetch_unmatched_summary(db=db, employee_id=employee_id, days=days)


# ---------------------------------------------------------------------------
# Single item explanation (explicit endpoint)
# ---------------------------------------------------------------------------

@router.get(
    "/items/{item_id}/explanation",
    response_model=AssistantExceptionItem,
)
async def explain_single_item(
    item_id: int,
    db: AsyncSession = Depends(get_db),
    # current_user=Depends(get_current_user),
):
    """
    Returns a bot-friendly summary of a single recon item.
    """
    return await _fetch_single_item(db=db, item_id=item_id)


# ---------------------------------------------------------------------------
# LLM-style query endpoint (intent router)
# ---------------------------------------------------------------------------

@router.post("/query", response_model=AssistantQueryResponse)
async def assistant_query(
    payload: AssistantQueryRequest,
    db: AsyncSession = Depends(get_db),
    # current_user=Depends(get_current_user),
):
    """
    One endpoint that tries to interpret the user's question and route it to:

    - FAQ answer
    - Exception summary (unmatched expenses by employee / date)
    - Single item explanation

    This is intentionally rule-based (no LLM) so that the Teams bot can call
    something deterministic from this service, and escalate to a real LLM
    only when 'kind' is 'unknown'.
    """
    query = payload.query.strip()
    if not query:
        return AssistantQueryResponse(
            kind="unknown",
            answer_text="I couldn't understand an empty question.",
        )

    q_lower = query.lower()

    # 1) Check if it's clearly about unmatched exceptions
    if "unmatched" in q_lower or "open items" in q_lower or "pending items" in q_lower:
        days = payload.days or _extract_days_from_text(query) or 7
        employee_id = payload.employee_id or _extract_employee_from_text(query)

        summary = await _fetch_unmatched_summary(
            db=db,
            employee_id=employee_id,
            days=days,
        )

        if employee_id:
            heading = f"Unmatched items for {employee_id} in the last {summary.days} days."
        else:
            heading = f"Unmatched items in the last {summary.days} days."

        extra = ""
        if summary.unmatched_count == 0:
            extra = " There are currently no unmatched items in this window."
        else:
            extra = (
                f" There are {summary.unmatched_count} unmatched items, "
                f"with an average age of {summary.avg_age_days:.1f} days."
            )

        answer_text = heading + extra

        return AssistantQueryResponse(
            kind="exception_summary",
            answer_text=answer_text,
            exception_summary=summary,
        )

    # 2) Check if it's about a specific item id
    item_id = _extract_item_id_from_text(query)
    if item_id is not None:
        item = await _fetch_single_item(db=db, item_id=item_id)

        answer_text = (
            f"Recon item {item.id} (TX {item.tx_id}) is currently {item.match_status}."
        )
        if item.reason:
            answer_text += f" Reason: {item.reason}."
        if item.sla_breached:
            answer_text += " This item has breached its SLA."

        return AssistantQueryResponse(
            kind="item_explanation",
            answer_text=answer_text,
            item=item,
        )

    # 3) Try FAQ-style answer
    faq = _match_best_faq(query)
    if faq:
        return AssistantQueryResponse(
            kind="faq",
            answer_text=faq.answer,
            faq=faq,
        )

    # 4) Fallback: unknown → let the higher-level LLM handle it
    return AssistantQueryResponse(
        kind="unknown",
        answer_text=(
            "I'm not sure how to answer that directly. "
            "You can ask about unmatched items, specific recon item IDs, or common reconciliation questions."
        ),
    )
