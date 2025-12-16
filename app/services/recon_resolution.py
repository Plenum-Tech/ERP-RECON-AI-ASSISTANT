# app/services/recon_resolution.py

from __future__ import annotations

import math
import re
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Dict, Optional, Literal

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import SQLAlchemyError

from app.models.recon import ReconItem

try:
    from app.utils.logger import get_logger

    log = get_logger(__name__)
except Exception:  # pragma: no cover
    import logging

    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Variance heuristics / suggested adjustment
# ---------------------------------------------------------------------------

def _norm(s: Any) -> str:
    """Lowercase, strip, collapse spaces; safe for None."""
    if s is None:
        return ""
    s = str(s)
    s = re.sub(r"\s+", " ", s)
    return s.strip().lower()


def _get_desc(statement_row: Dict[str, Any]) -> str:
    """
    Try to pull a meaningful description from the normalized statement row.
    We look at common BC / bank headers first, then fall back to raw_desc.
    """
    raw = statement_row.get("Description") or statement_row.get("description")
    if not raw:
        raw = statement_row.get("raw_desc")
    if not raw:
        # maybe inside .get("raw") if not normalized yet
        raw_payload = statement_row.get("raw") or {}
        raw = (
            raw_payload.get("Description")
            or raw_payload.get("description")
            or raw_payload.get("Narration")
            or raw_payload.get("memo")
        )
    return str(raw or "")


def _detect_fx(desc_norm: str, currency_code: Optional[str]) -> bool:
    if currency_code and _norm(currency_code) not in {"aed", ""}:
        # Non-AED → very likely FX-related in this context
        return True

    fx_keywords = [
        "fx",
        "foreign exchange",
        "exchange rate",
        "forex",
        "currency conversion",
        "conversion fee",
        "rate diff",
        "exchange diff",
        "fx diff",
    ]
    return any(k in desc_norm for k in fx_keywords)


def _detect_fee_or_tip(desc_norm: str) -> bool:
    fee_kw = [
        "fee",
        "fees",
        "charge",
        "charges",
        "commission",
        "bank charge",
        "bank charges",
        "service fee",
        "service charge",
        "processing fee",
    ]
    tip_kw = [
        "tip",
        "gratuity",
        "gratuities",
    ]
    return any(k in desc_norm for k in fee_kw + tip_kw)


def _detect_deposit_or_salary(desc_norm: str) -> bool:
    dep_kw = [
        "deposit",
        "salary",
        "payroll",
        "wage",
        "wages",
        "salary transfer",
        "salary credit",
        "credit salary",
    ]
    return any(k in desc_norm for k in dep_kw)


def _safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float, Decimal)):
        return float(x)
    try:
        return float(str(x).replace(",", ""))
    except Exception:
        return None


def _proposed_journal_stub(
    category: str,
    variance_amount: float,
    currency: str,
) -> Dict[str, Any]:
    """
    Conservative stub for proposed journal entry. We DO NOT assume real GL
    account codes here; we only give hints / labels. Caller can map to
    actual accounts in ERP config.
    """
    amt_abs = abs(variance_amount)
    base = {
        "amount": amt_abs,
        "currency": currency,
        "note": "Template only; map 'account_label' to real GL accounts in ERP.",
    }

    if amt_abs <= 0:
        return {}

    if category == "fx_variance":
        return {
            **base,
            "lines": [
                {"side": "DR", "account_label": "FX Loss / Gain", "amount": amt_abs},
                {"side": "CR", "account_label": "Bank / Clearing", "amount": amt_abs},
            ],
        }

    if category == "bank_fee_or_tip":
        return {
            **base,
            "lines": [
                {"side": "DR", "account_label": "Bank Charges / Tips Expense", "amount": amt_abs},
                {"side": "CR", "account_label": "Bank", "amount": amt_abs},
            ],
        }

    if category == "partial_payment_or_shortpay":
        return {
            **base,
            "lines": [
                {
                    "side": "DR",
                    "account_label": "Short Payment / Write-off",
                    "amount": amt_abs,
                },
                {"side": "CR", "account_label": "Customer / Receivable", "amount": amt_abs},
            ],
        }

    # generic fallback
    return {
        **base,
        "lines": [
            {"side": "DR", "account_label": "Adjustment", "amount": amt_abs},
            {"side": "CR", "account_label": "Bank / Clearing", "amount": amt_abs},
        ],
    }


def build_suggested_adjustment(
    *,
    tx_amount: Optional[float],
    variance_amount: Optional[float],
    statement_row: Dict[str, Any],
    match_info: Optional[Dict[str, Any]],
    reco_info: Optional[Dict[str, Any]],
    currency: str = "AED",
) -> Optional[Dict[str, Any]]:
    """
    Heuristic variance classifier for a single ReconItem candidate.

    Returns a JSON-serializable dict like:
    {
      "category": "fx_variance" | "bank_fee_or_tip" | "partial_payment_or_shortpay" | "other",
      "confidence": 0.0 .. 1.0,
      "reason": "...",
      "amount": <absolute variance>,
      "currency": "AED",
      "hints": { ... },
      "proposed_journal": { ... }  # purely a template, no real GL codes
    }

    If we cannot confidently classify / no real variance, returns None.
    """
    try:
        # No variance / trivial → no suggestion
        if variance_amount is None:
            return None

        var = _safe_float(variance_amount)
        if var is None:
            return None

        var_abs = abs(var)
        if var_abs < 0.01:
            # below 1-cent equivalent; ignore
            return None

        desc = _get_desc(statement_row)
        desc_norm = _norm(desc)

        raw = statement_row.get("raw") or {}
        currency_code = (
            raw.get("Currency Code")
            or raw.get("currency_code")
            or raw.get("Currency")
            or currency
        )

        # ------------------------------------------------------------------ #
        # 1) FX variance
        # ------------------------------------------------------------------ #
        if _detect_fx(desc_norm, currency_code):
            reason = (
                f"Variance looks FX-related: currency={currency_code!r}, "
                f"description contains FX-like wording."
            )
            category = "fx_variance"
            confidence = 0.9
            return {
                "category": category,
                "confidence": round(confidence, 3),
                "reason": reason,
                "amount": var_abs,
                "currency": currency_code or currency,
                "hints": {
                    "statement_description": desc,
                    "statement_row_id": statement_row.get("row_id"),
                    "match_type": (match_info or {}).get("type"),
                },
                "proposed_journal": _proposed_journal_stub(category, var, currency_code or currency),
            }

        # ------------------------------------------------------------------ #
        # 2) Bank fee / tip / charges
        # ------------------------------------------------------------------ #
        if _detect_fee_or_tip(desc_norm):
            reason = (
                "Description indicates fees/charges/tips; "
                "variance likely due to bank charges or gratuity."
            )
            category = "bank_fee_or_tip"
            confidence = 0.85
            return {
                "category": category,
                "confidence": round(confidence, 3),
                "reason": reason,
                "amount": var_abs,
                "currency": currency_code or currency,
                "hints": {
                    "statement_description": desc,
                    "statement_row_id": statement_row.get("row_id"),
                    "match_type": (match_info or {}).get("type"),
                },
                "proposed_journal": _proposed_journal_stub(category, var, currency_code or currency),
            }

        # ------------------------------------------------------------------ #
        # 3) Partial payment / short-pay (VERY conservative)
        # ------------------------------------------------------------------ #
        # If we have a match, description looks like deposit/salary,
        # and variance is relatively small (e.g. < 10% of tx_amount),
        # treat as probable partial / short payment.
        tx = _safe_float(tx_amount) or 0.0
        rel_gap = (var_abs / abs(tx)) if tx not in {0.0, None} else None

        if match_info and _detect_deposit_or_salary(desc_norm) and rel_gap is not None:
            if 0.0 < rel_gap <= 0.1:
                reason = (
                    "Matched deposit/salary with small residual variance; "
                    "likely a short-pay / partial payment / rounding difference."
                )
                category = "partial_payment_or_shortpay"
                confidence = 0.75
                return {
                    "category": category,
                    "confidence": round(confidence, 3),
                    "reason": reason,
                    "amount": var_abs,
                    "currency": currency_code or currency,
                    "hints": {
                        "statement_description": desc,
                        "statement_row_id": statement_row.get("row_id"),
                        "match_type": (match_info or {}).get("type"),
                        "relative_gap_pct": round(rel_gap * 100, 4),
                    },
                    "proposed_journal": _proposed_journal_stub(
                        category, var, currency_code or currency
                    ),
                }

        # ------------------------------------------------------------------ #
        # 4) Fallback "other" (only if variance is non-trivial)
        # ------------------------------------------------------------------ #
        reason = (
            "Variance detected but does not clearly match FX, fee/tip, or short-pay patterns. "
            "Treat as generic recon variance for manual review."
        )
        category = "other"
        confidence = 0.4

        return {
            "category": category,
            "confidence": round(confidence, 3),
            "reason": reason,
            "amount": var_abs,
            "currency": currency_code or currency,
            "hints": {
                "statement_description": desc,
                "statement_row_id": statement_row.get("row_id"),
                "match_type": (match_info or {}).get("type") if match_info else None,
            },
            "proposed_journal": _proposed_journal_stub(category, var, currency_code or currency),
        }

    except Exception as e:  # pragma: no cover
        log.warning("build_suggested_adjustment failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# Resolution (apply / ignore) + SLA handling
# ---------------------------------------------------------------------------

class ReconResolutionError(Exception):
    """Raised when a recon item cannot be resolved due to business rules."""

    pass


class ReconItemNotFound(Exception):
    """Raised when a recon item with the given id does not exist."""

    pass


async def resolve_recon_item(
    db: AsyncSession,
    *,
    item_id: int,
    action: Literal["apply", "ignore"],
    note: Optional[str] = None,
    actor_id: Optional[str] = None,
) -> ReconItem:
    """
    Resolve a ReconItem as 'apply' or 'ignore'.

    Semantics (v1):
    - Prevent double-resolution if the item is already applied/ignored.
    - Mark it closed (closed_at) and update SLA breach flag if sla_due_at is present.
    - Optionally populate resolution audit fields if present on the ORM model:
        - resolution_status
        - resolution_note
        - resolved_by
        - resolved_at
    - Optionally set owner_id to actor_id if owner is empty.
    """

    # 1. Load item
    stmt = select(ReconItem).where(ReconItem.id == item_id)
    result = await db.execute(stmt)
    item: Optional[ReconItem] = result.scalar_one_or_none()

    if item is None:
        raise ReconItemNotFound(f"ReconItem {item_id} not found")

    # 2. Guard: avoid double-resolution if model has resolution_status
    existing_status: Optional[str] = None
    if hasattr(item, "resolution_status"):
        existing_status = getattr(item, "resolution_status", None)

    if existing_status in {"applied", "ignored"}:
        raise ReconResolutionError(
            f"ReconItem {item_id} is already resolved as '{existing_status}'"
        )

    # 3. Apply resolution semantics
    now = datetime.utcnow()

    # resolution_status
    if hasattr(item, "resolution_status"):
        setattr(item, "resolution_status", "applied" if action == "apply" else "ignored")

    # resolution_note
    if hasattr(item, "resolution_note"):
        if note is not None:
            setattr(item, "resolution_note", note)

    # resolved_by
    if hasattr(item, "resolved_by") and actor_id is not None:
        setattr(item, "resolved_by", actor_id)

    # resolved_at
    if hasattr(item, "resolved_at"):
        setattr(item, "resolved_at", now)

    # closed_at
    if hasattr(item, "closed_at"):
        if getattr(item, "closed_at", None) is None:
            setattr(item, "closed_at", now)

    # set owner_id if missing
    if actor_id is not None and hasattr(item, "owner_id"):
        if getattr(item, "owner_id", None) is None:
            setattr(item, "owner_id", actor_id)

    # 4. SLA breach: sla_due_at / sla_breached (if present)
    sla_due_at = getattr(item, "sla_due_at", None) if hasattr(item, "sla_due_at") else None
    if sla_due_at is not None and hasattr(item, "sla_breached"):
        breached = now > sla_due_at
        setattr(item, "sla_breached", breached)

    # 5. Persist
    try:
        await db.flush()
        await db.commit()
        await db.refresh(item)
    except SQLAlchemyError as e:
        await db.rollback()
        log.exception("Failed to resolve ReconItem %s: %s", item_id, e)
        raise ReconResolutionError(
            f"Failed to resolve ReconItem {item_id}: {e}"
        ) from e

    return item
