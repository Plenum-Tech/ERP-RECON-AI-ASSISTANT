# app/services/variance_reasoner.py

from __future__ import annotations

import math
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from app.utils.logger import get_logger

    log = get_logger(__name__)
except Exception:  # pragma: no cover
    import logging

    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(__name__)


def _canon(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def _has_any(text: str, keywords: Iterable[str]) -> bool:
    t = _canon(text)
    return any(k in t for k in keywords)


def _safe_abs(x: Optional[float]) -> float:
    try:
        return float(abs(x)) if x is not None else 0.0
    except Exception:
        return 0.0


def is_within_tolerance(
    tx_amount: Optional[float],
    variance_amount: Optional[float],
    tol_abs: float,
    tol_pct: float,
) -> bool:
    """
    Decide if a variance is "small enough" to be safely auto-adjustable.

    Uses:
    - absolute threshold: tol_abs
    - relative threshold: tol_pct * |tx_amount|
    """
    v = _safe_abs(variance_amount)
    t = _safe_abs(tx_amount)

    # absolute rule
    if v <= tol_abs:
        return True

    # relative rule (if we know tx_amount != 0)
    if t > 0:
        return (v / t) <= tol_pct

    return False


def classify_variance_reason(
    *,
    tx_amount: Optional[float],
    variance_amount: Optional[float],
    currency: str = "AED",
    statement_desc: str = "",
    ledger_descs: Optional[List[str]] = None,
) -> str:
    """
    Heuristic classifier for variance reason.

    Returns one of:
    - "none"            : effectively no variance
    - "fx_rounding"     : very small variance → likely FX or rounding
    - "tip_variance"    : tips / gratuity / service charge
    - "fee_variance"    : bank fees / charges / commissions
    - "partial_payment" : line appears to be a partial settle
    - "timing_difference": looks like timing/period issue
    - "other"           : generic bucket
    """
    v = _safe_abs(variance_amount)
    t = _safe_abs(tx_amount)
    all_desc = statement_desc or ""
    if ledger_descs:
        all_desc += " || " + " || ".join(ld or "" for ld in ledger_descs)

    # 0) Effectively no variance
    if v < 1e-6:
        return "none"

    # 1) Very small variance → FX / rounding
    if v <= 1.0:
        # up to 1 unit in currency → call it FX/rounding
        return "fx_rounding"

    # 2) Tip-related keywords
    if _has_any(all_desc, ["tip", "gratuity", "service charge", "svc charge"]):
        return "tip_variance"

    # 3) Fee / bank charges
    if _has_any(
        all_desc,
        ["fee", "fees", "charges", "bank charge", "commission", "processing fee"],
    ):
        return "fee_variance"

    # 4) Potential partial payment
    # If variance is a big chunk (e.g. >30% of transaction) and
    # wording suggests "installment" / "partial".
    if t > 0 and (v / t) >= 0.3:
        if _has_any(
            all_desc,
            ["partial", "instalment", "installment", "part payment", "advance"],
        ):
            return "partial_payment"

    # 5) Timing diff hints
    if _has_any(
        all_desc,
        ["backdated", "prior period", "cutoff", "carry forward", "c/f", "b/f"],
    ):
        return "timing_difference"

    # Default
    return "other"


def build_suggested_adjustment(
    *,
    tx_amount: Optional[float],
    variance_amount: Optional[float],
    currency: str,
    variance_reason: str,
    statement_desc: str = "",
    ledger_descs: Optional[List[str]] = None,
    auto_apply_safe: bool,
) -> Optional[Dict[str, Any]]:
    """
    Construct a structured suggested_adjustment JSON blob.

    Shape:

    {
      "auto_apply_safe": true/false,
      "reason": "fx_rounding" | "tip_variance" | ...,
      "amount": 1.23,
      "currency": "AED",
      "lines": [
        {
          "account_hint": "Bank",
          "direction": "debit" | "credit",
          "amount": 1.23,
          "description": "...",
        },
        {
          "account_hint": "FX Gain/Loss",
          "direction": "credit",
          "amount": 1.23,
          "description": "...",
        }
      ],
      "notes": "...free text..."
    }

    Returns None if we do not want to suggest anything.
    """
    v = _safe_abs(variance_amount)
    if v <= 0.0:
        return None

    # If we don't deem it safe, don't construct lines; just metadata.
    if not auto_apply_safe:
        return {
            "auto_apply_safe": False,
            "reason": variance_reason,
            "amount": v,
            "currency": currency,
            "lines": [],
            "notes": "Variance not within auto-apply tolerance.",
        }

    # Decide template based on reason
    lines: List[Dict[str, Any]] = []
    notes: List[str] = []

    base_desc = statement_desc or ""
    if ledger_descs:
        base_desc += " | " + " | ".join(ledger_descs)

    if variance_reason in {"fx_rounding", "fx_variance"}:
        # Bank vs FX Gain/Loss
        lines.append(
            {
                "account_hint": "Bank",
                "direction": "debit" if (variance_amount or 0.0) > 0 else "credit",
                "amount": v,
                "description": f"FX rounding adjustment ({currency})",
            }
        )
        lines.append(
            {
                "account_hint": "FX Gain/Loss",
                "direction": "credit" if (variance_amount or 0.0) > 0 else "debit",
                "amount": v,
                "description": f"Offset FX rounding ({currency})",
            }
        )
        notes.append("Suggest posting FX rounding difference to FX Gain/Loss.")

    elif variance_reason == "tip_variance":
        lines.append(
            {
                "account_hint": "Bank",
                "direction": "debit" if (variance_amount or 0.0) > 0 else "credit",
                "amount": v,
                "description": "Tip variance on POS transaction",
            }
        )
        lines.append(
            {
                "account_hint": "Tips/Gratuity Expense",
                "direction": "credit" if (variance_amount or 0.0) > 0 else "debit",
                "amount": v,
                "description": "Offset tip variance",
            }
        )
        notes.append("Suggest posting tip difference to Tips/Gratuity Expense.")

    elif variance_reason == "fee_variance":
        lines.append(
            {
                "account_hint": "Bank",
                "direction": "debit" if (variance_amount or 0.0) > 0 else "credit",
                "amount": v,
                "description": "Bank fee / charges variance",
            }
        )
        lines.append(
            {
                "account_hint": "Bank Charges Expense",
                "direction": "credit" if (variance_amount or 0.0) > 0 else "debit",
                "amount": v,
                "description": "Offset bank fee variance",
            }
        )
        notes.append("Suggest posting fee difference to Bank Charges Expense.")

    elif variance_reason in {"partial_payment", "timing_difference"}:
        # We do NOT auto-build lines; we just flag it.
        notes.append(
            "Variance looks like partial payment or timing difference; "
            "manual review recommended."
        )
        return {
            "auto_apply_safe": False,
            "reason": variance_reason,
            "amount": v,
            "currency": currency,
            "lines": [],
            "notes": " ".join(notes),
        }

    else:
        # generic other small variance – we flag but keep safe=false to avoid surprises
        return {
            "auto_apply_safe": False,
            "reason": variance_reason,
            "amount": v,
            "currency": currency,
            "lines": [],
            "notes": "Variance classified as 'other'; manual review required.",
        }

    return {
        "auto_apply_safe": auto_apply_safe,
        "reason": variance_reason,
        "amount": v,
        "currency": currency,
        "lines": lines,
        "notes": " ".join(notes),
        "context": {"source_desc": base_desc[:500]},
    }


def compute_variance_metadata(
    *,
    tx_amount: Optional[float],
    variance_amount: Optional[float],
    currency: str,
    statement_desc: str,
    ledger_descs: Optional[List[str]],
    tol_abs: float,
    tol_pct: float,
) -> Dict[str, Any]:
    """
    Convenience helper to compute:
    - variance_reason (string)
    - auto_apply_safe (bool)
    - suggested_adjustment (dict / None)
    - plus a small diagnostic payload you can merge into variance_payload.
    """
    variance_reason = classify_variance_reason(
        tx_amount=tx_amount,
        variance_amount=variance_amount,
        currency=currency,
        statement_desc=statement_desc,
        ledger_descs=ledger_descs,
    )

    auto_safe = is_within_tolerance(
        tx_amount=tx_amount,
        variance_amount=variance_amount,
        tol_abs=tol_abs,
        tol_pct=tol_pct,
    )

    suggested = build_suggested_adjustment(
        tx_amount=tx_amount,
        variance_amount=variance_amount,
        currency=currency,
        variance_reason=variance_reason,
        statement_desc=statement_desc,
        ledger_descs=ledger_descs,
        auto_apply_safe=auto_safe,
    )

    meta = {
        "variance_reason": variance_reason,
        "auto_apply_safe": auto_safe,
        "suggested_adjustment": suggested,
    }

    # Small diagnostic you can embed in variance_payload
    diagnostic = {
        "variance_reason": variance_reason,
        "auto_apply_safe": auto_safe,
        "tolerance": {"abs": tol_abs, "pct": tol_pct},
    }

    meta["diagnostic"] = diagnostic
    return meta
