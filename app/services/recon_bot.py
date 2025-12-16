# app/services/recon_bot.py

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.recon import ReconItem
from app.schemas.recon_bot import ReconBotReply, ReconExceptionItem
from app.services.llm_client import (
    LLM_AVAILABLE,
    classify_recon_bot_intent,
    draft_exception_lookup_reply,
    answer_faq_freestyle,
)

try:
    from app.utils.logger import get_logger

    log = get_logger(__name__)
except Exception:  # pragma: no cover
    import logging

    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _ensure_naive_utc(dt: Optional[datetime]) -> Optional[datetime]:
    """
    Normalize any datetime to a naive UTC datetime.

    - If dt is None: return None.
    - If tz-aware: convert to UTC, then drop tzinfo.
    - If already naive: return as-is.
    """
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt
    return dt.astimezone(timezone.utc).replace(tzinfo=None)


async def _query_exceptions(
    db: AsyncSession,
    *,
    employee_name: Optional[str],
    window_days: int,
    status: Optional[str],
    max_items: int = 50,
) -> List[ReconItem]:
    """
    Query ReconItem for exceptions based on simple filters.

    - Filters by opened_at in the last `window_days`.
    - If status is provided, filters by match_status.
    - If employee_name is provided, performs a case-insensitive LIKE
      on employee_id (crude, but fine for v1).
    """
    now = _ensure_naive_utc(datetime.utcnow())
    since = now - timedelta(days=window_days)  # type: ignore[operator]

    stmt = select(ReconItem).where(ReconItem.opened_at >= since)

    # status filter (default unmatched if provided by caller)
    if status:
        stmt = stmt.where(ReconItem.match_status == status)

    # crude employee_name filter: match against employee_id using ILIKE
    if employee_name:
        stmt = stmt.where(
            func.lower(ReconItem.employee_id).like(f"%{employee_name.lower()}%")
        )

    stmt = stmt.order_by(ReconItem.opened_at.desc()).limit(max_items)
    result = await db.execute(stmt)
    return result.scalars().all()


def _extract_variance_reason(item: ReconItem) -> Optional[str]:
    """
    Try to pull a human-friendly variance reason from variance_payload.

    When we stored variance_payload in bank_reco_store, we added:
      - variance_reason
      - auto_apply_safe
      - variance_meta (diagnostic)
    """
    payload = item.variance_payload or {}
    reason = payload.get("variance_reason") or payload.get("reason")
    if isinstance(reason, str):
        return reason
    return None


def _map_item_to_exception_schema(item: ReconItem) -> ReconExceptionItem:
    """
    Map ReconItem ORM row to a ReconExceptionItem schema for the bot.
    """
    return ReconExceptionItem(
        id=item.id,
        tx_id=item.tx_id,
        employee_id=item.employee_id,
        owner_id=item.owner_id,
        match_status=item.match_status,
        variance_type=item.variance_type,
        variance_amount=item.variance_amount,
        sla_breached=item.sla_breached,
        opened_at=item.opened_at,
        variance_reason=_extract_variance_reason(item),
        # Deep link is optional; you can add it later via a builder
        deep_link=None,
    )


def _summarise_items_for_llm(
    *,
    items: List[ReconItem],
    employee_name: Optional[str],
    window_days: int,
    status: Optional[str],
) -> Dict[str, Any]:
    """
    Build a JSON summary for the LLM to turn into text.
    """
    now = _ensure_naive_utc(datetime.utcnow())

    def _age_days(opened_at: Optional[datetime]) -> float:
        opened_at_norm = _ensure_naive_utc(opened_at)
        if not opened_at_norm or not now:
            return 0.0
        delta = now - opened_at_norm
        return round(delta.total_seconds() / 86400.0, 2)

    summary_items: List[Dict[str, Any]] = []
    for i in items[:5]:  # top 5 only for summary
        opened_norm = _ensure_naive_utc(i.opened_at) if i.opened_at else None
        summary_items.append(
            {
                "id": i.id,
                "tx_id": i.tx_id,
                "employee_id": i.employee_id,
                "owner_id": i.owner_id,
                "currency": i.currency,
                "tx_amount": float(i.tx_amount) if i.tx_amount is not None else None,
                "variance_type": i.variance_type,
                "variance_amount": float(i.variance_amount)
                if i.variance_amount is not None
                else None,
                "match_status": i.match_status,
                "sla_breached": i.sla_breached,
                "opened_at": opened_norm.isoformat() if opened_norm else None,
                "resolution_status": i.resolution_status,
                "variance_reason": _extract_variance_reason(i),
                "age_days": _age_days(i.opened_at),
            }
        )

    return {
        "employee_name": employee_name,
        "window_days": window_days,
        "status_filter": status,
        "total_count": len(items),
        "items_shown": len(summary_items),
        "items": summary_items,
    }


# ---------------------------------------------------------------------
# Main bot handler
# ---------------------------------------------------------------------


async def handle_bot_message(
    db: AsyncSession,
    *,
    message: str,
    user_id: str,
    org_id: Optional[str] = None,
) -> ReconBotReply:
    """
    Main entry point for the Recon bot.

    - If LLM is not configured, return a friendly message + metadata.
    - Otherwise:
      - Classify intent via LLM
      - Route to:
        - FAQ answer (LLM-only for now)
        - Exception lookup (DB + LLM summary)
        - Fallback generic response

    Returns a ReconBotReply Pydantic model, with:
    - reply_text: what to show in Teams
    - intent: "exception_lookup" | "faq" | "other"
    - parameters: raw intent parameters from the classifier
    - exceptions: list of ReconExceptionItem for UI
    """
    if not LLM_AVAILABLE:
        # Graceful fallback instead of crashing the app
        return ReconBotReply(
            reply_text=(
                "Recon bot is not configured with an LLM API key yet. "
                "Please ask your admin to set OPENAI_API_KEY or settings.openai_api_key."
            ),
            intent="other",
            parameters={},
            exceptions=[],
        )

    try:
        # 1) Classify intent
        try:
            intent_payload = await classify_recon_bot_intent(message)
        except Exception as e:
            log.exception("Intent classification LLM error: %s", e)
            return ReconBotReply(
                reply_text=(
                    "I couldn't classify that request just now. "
                    "Please try rephrasing or try again in a moment."
                ),
                intent="other",
                parameters={},
                exceptions=[],
            )

        intent = intent_payload.get("intent", "other")
        params = intent_payload.get("parameters", {}) or {}

        log.info("Recon bot intent: %s params=%s", intent, params)

        raw_question = params.get("raw_question") or message

        # -------------------- FAQ --------------------
        if intent == "faq":
            try:
                answer = await answer_faq_freestyle(raw_question)
            except Exception as e:
                log.exception("FAQ LLM error: %s", e)
                answer = "Sorry, I couldn't reach the FAQ assistant just now."

            return ReconBotReply(
                reply_text=answer,
                intent=intent,
                parameters=params,
                exceptions=[],
            )

        # ----------------- Exception lookup -----------------
        if intent == "exception_lookup":
            employee_name = params.get("employee_name") or None
            window_days = params.get("window_days") or 7
            try:
                window_days = int(window_days)
            except Exception:
                window_days = 7

            # Clamp window to something sane
            if window_days <= 0:
                window_days = 1
            if window_days > 90:
                window_days = 90

            status = params.get("status") or "unmatched"

            items = await _query_exceptions(
                db=db,
                employee_name=employee_name,
                window_days=window_days,
                status=status,
                max_items=50,
            )

            summary = _summarise_items_for_llm(
                items=items,
                employee_name=employee_name,
                window_days=window_days,
                status=status,
            )

            # Top few exceptions to attach to the reply
            exception_schemas = [
                _map_item_to_exception_schema(i) for i in items[:5]
            ]

            # Let LLM write the user-facing text
            try:
                reply_text = await draft_exception_lookup_reply(summary)
            except Exception as e:
                log.exception("Exception lookup LLM error: %s", e)
                if not items:
                    reply_text = (
                        f"No {status or 'unmatched'} items found for "
                        f"{employee_name or 'the selected employee'} "
                        f"in the last {window_days} day(s)."
                    )
                else:
                    reply_text = (
                        f"Found {len(items)} {status or 'unmatched'} item(s) "
                        f"for {employee_name or 'the selected employee'} "
                        f"in the last {window_days} day(s)."
                    )

            return ReconBotReply(
                reply_text=reply_text,
                intent=intent,
                parameters=params,
                exceptions=exception_schemas,
            )

        # ----------------- Fallback / other -----------------
        fallback_text = (
            "I can help with:\n"
            "- Recon FAQs (FX variances, tips, partial matches, SLA, etc.)\n"
            "- Exception lookups like: 'Show unmatched expenses for Ahmed last 7 days'."
        )

        return ReconBotReply(
            reply_text=fallback_text,
            intent="other",
            parameters=params,
            exceptions=[],
        )

    except Exception as e:
        # Final safety net so the bot never throws a 500
        log.exception("Recon bot error for user %s: %s", user_id, e)
        return ReconBotReply(
            reply_text=(
                "Something went wrong while checking reconciliations. "
                "Please try again in a bit or contact the finance ops team."
            ),
            intent="other",
            parameters={},
            exceptions=[],
        )
