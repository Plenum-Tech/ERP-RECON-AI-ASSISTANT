from __future__ import annotations

from typing import Any, Dict, List, Optional
from datetime import date, datetime

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import SQLAlchemyError

from app.models.recon import ReconItem  # your ReconItem model
from app.services.variance_reasoner import compute_variance_metadata
from app.services.recon_sla import apply_default_sla  # ✅ use existing helper

# Logger (same pattern as other modules)
try:
    from app.utils.logger import get_logger

    log = get_logger(__name__)
except Exception:  # pragma: no cover
    import logging

    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(__name__)


def _safe_get(arr: List[Dict[str, Any]], idx: int) -> Dict[str, Any]:
    """Safe index helper."""
    try:
        return arr[idx]
    except Exception:
        return {}


def _to_json_safe(obj: Any) -> Any:
    """
    Recursively convert an object into something JSON-serializable.

    - datetime / date / pandas.Timestamp -> ISO string
    - dict -> dict with json-safe values
    - list/tuple/set -> list of json-safe values
    - everything else -> keep if primitive, else str(obj)
    """
    # Primitive types are already JSON-safe
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj

    # Date/time types (datetime, date, pandas.Timestamp, etc.)
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()

    # Anything with an isoformat method (e.g., pandas.Timestamp)
    if hasattr(obj, "isoformat"):
        try:
            return obj.isoformat()
        except Exception:
            pass

    # Dict -> recurse
    if isinstance(obj, dict):
        return {str(k): _to_json_safe(v) for k, v in obj.items()}

    # List / tuple / set -> recurse
    if isinstance(obj, (list, tuple, set)):
        return [_to_json_safe(v) for v in obj]

    # Fallback: string representation
    return str(obj)


def _index_matches_by_line(matcher_result: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    """
    Build an index: line_index -> {type, matches: [..]} from matcher_result["matches"].
    """
    matches = matcher_result.get("matches", {}) or {}

    one_to_one = matches.get("one_to_one", []) or []
    comp_line_from_ledger = matches.get("composite_line_from_ledger", []) or []
    comp_ledger_from_lines = matches.get("composite_ledger_from_lines", []) or []

    by_line: Dict[int, Dict[str, Any]] = {}

    # 1-to-1
    for m in one_to_one:
        li = int(m["line_index"])
        bucket = by_line.setdefault(li, {"matches": [], "types": set()})
        bucket["matches"].append(m)
        bucket["types"].add("1to1")

    # composite: statement line = ledger1 + ledger2 + ...
    for m in comp_line_from_ledger:
        li = int(m["line_index"])
        bucket = by_line.setdefault(li, {"matches": [], "types": set()})
        bucket["matches"].append(m)
        bucket["types"].add("composite_line_from_ledger")

    # composite: ledger line = line1 + line2 + ...
    # Here lines are in "line_indices"; we still want to tag each statement line.
    for m in comp_ledger_from_lines:
        for li in m.get("line_indices", []) or []:
            li = int(li)
            bucket = by_line.setdefault(li, {"matches": [], "types": set()})
            bucket["matches"].append(m)
            bucket["types"].add("composite_ledger_from_lines")

    # Normalize types -> single string where obvious, else "mixed"
    for li, info in by_line.items():
        types = info["types"]
        if len(types) == 1:
            info["type"] = list(types)[0]
        else:
            info["type"] = "mixed"

    return by_line


def _index_recos_by_line(matcher_result: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    """
    Build: line_index -> rec_entry from matcher_result["recommendations"].
    """
    recos = matcher_result.get("recommendations", []) or []
    out: Dict[int, Dict[str, Any]] = {}

    def best_score(rec: Dict[str, Any]) -> float:
        cands = rec.get("candidates", []) or []
        if not cands:
            return 0.0
        return float(cands[0].get("score", 0.0))

    for r in recos:
        li = int(r["line_index"])
        existing = out.get(li)
        if not existing:
            out[li] = r
        else:
            # keep the recommendation whose top candidate has the highest score
            if best_score(r) > best_score(existing):
                out[li] = r
    return out


def _collect_ledger_descs_for_line(
    match_info: Optional[Dict[str, Any]],
    ledger_rows: List[Dict[str, Any]],
) -> List[str]:
    """
    From match_info["matches"], collect ledger raw_desc strings for this line.
    Handles:
    - 1to1: m["ledger_index"]
    - composite_line_from_ledger: m["ledger_indices"]
    - composite_ledger_from_lines: same keys present
    """
    if not match_info:
        return []

    descs: List[str] = []
    matches = match_info.get("matches", []) or []

    for m in matches:
        # 1-to-1 case
        if "ledger_index" in m:
            idx = m.get("ledger_index")
            try:
                j = int(idx)
            except (TypeError, ValueError):
                j = None
            if j is not None and 0 <= j < len(ledger_rows):
                d = ledger_rows[j].get("raw_desc")
                if d:
                    descs.append(str(d))

        # composite cases
        for idx in m.get("ledger_indices", []) or []:
            try:
                j = int(idx)
            except (TypeError, ValueError):
                continue
            if 0 <= j < len(ledger_rows):
                d = ledger_rows[j].get("raw_desc")
                if d:
                    descs.append(str(d))

    # Deduplicate while preserving order
    seen = set()
    uniq: List[str] = []
    for d in descs:
        if d not in seen:
            seen.add(d)
            uniq.append(d)

    return uniq


async def store_bank_reco_result(
    db: AsyncSession,
    *,
    matcher_result: Dict[str, Any],
    org_id: Optional[str] = None,
    user_id: Optional[str] = None,
    default_currency: str = "AED",
    source_system: str = "BANK",
    tx_id_map: Optional[Dict[int, int]] = None,
    run_meta: Optional[Dict[str, Any]] = None,
) -> List[ReconItem]:
    """
    Persist a bank reconciliation result into erp_ai.recon_items (async).

    - If tx_id_map is provided, tx_id is taken from that mapping.
    - If tx_id_map is None, we fall back to using stmt_row_id (non-null)
      so that the NOT NULL constraint on tx_id is satisfied.

    Now also:
    - Computes variance_reason (fx_rounding / tip_variance / fee_variance / partial / etc.)
    - Computes auto_apply_safe based on tol_abs / tol_pct
    - Populates suggested_adjustment JSON for safe cases.
    """

    normalized_full = matcher_result.get("normalized_full", {}) or {}
    statement_rows: List[Dict[str, Any]] = normalized_full.get("statement") or []
    ledger_rows: List[Dict[str, Any]] = normalized_full.get("ledger") or []

    matches_by_line = _index_matches_by_line(matcher_result)
    recos_by_line = _index_recos_by_line(matcher_result)

    created_items: List[ReconItem] = []

    # Extract tolerance from run_meta for consistency with matcher
    run_meta_safe = run_meta or {}
    tol_abs = float(run_meta_safe.get("tol_abs", 2.0))
    tol_pct = float(run_meta_safe.get("tol_pct", 0.005))

    # Use a single "now" for all items in this run
    now = datetime.utcnow()

    try:
        for line_index, stmt in enumerate(statement_rows):
            # line_index here is the position in normalized statement list
            stmt_row_id = int(stmt.get("row_id", line_index))
            tx_amount = float(stmt.get("signed_amount") or 0.0)

            # currency: try find in raw, fallback to default
            raw = stmt.get("raw", {}) or {}
            currency = (
                raw.get("currency")
                or raw.get("Currency")
                or raw.get("currency_code")
                or raw.get("Currency Code")
                or default_currency
            )

            # tx_id mapping:
            # - if tx_id_map is provided: use it
            # - else: fall back to stmt_row_id so tx_id is never NULL
            if tx_id_map is not None:
                tx_id_val = tx_id_map.get(stmt_row_id)
                # if not found in map, still fall back to stmt_row_id
                if tx_id_val is None:
                    tx_id_val = stmt_row_id
            else:
                tx_id_val = stmt_row_id

            match_info = matches_by_line.get(line_index)
            reco_info = recos_by_line.get(line_index)

            if match_info:
                match_type = match_info.get("type", "matched")
                match_status = "matched"
                variance_type = f"bank_gl_{match_type}"
            else:
                match_type = None
                match_status = "unmatched"
                variance_type = "bank_gl_unmatched"

            # Compute variance_amount (absolute)
            variance_amount: Optional[float] = None

            if match_info and match_type == "1to1":
                # first 1-to-1 match
                m0 = next(
                    (m for m in match_info["matches"] if m.get("type") == "1to1"),
                    match_info["matches"][0],
                )
                led_amt = float(m0.get("ledger_amount") or 0.0)
                variance_amount = abs(tx_amount - led_amt)

            elif match_info and match_type == "composite_line_from_ledger":
                # statement line amount vs sum of ledger_amounts
                m0 = match_info["matches"][0]
                led_amts = [float(x) for x in (m0.get("ledger_amounts") or [])]
                total_led = sum(led_amts)
                variance_amount = abs(tx_amount - total_led)

            elif match_info and match_type == "composite_ledger_from_lines":
                # group: ledger line = sum(statement line_amounts)
                # We don't know each line's share of the variance,
                # so we compute group-level variance and reuse it.
                m0 = match_info["matches"][0]
                line_amts = [float(x) for x in (m0.get("line_amounts") or [])]
                ledger_amt = float(m0.get("ledger_amount") or 0.0)
                total_lines = sum(line_amts)
                variance_amount = abs(total_lines - ledger_amt)

            else:
                # unmatched or unknown type: variance is the full amount as a flag
                variance_amount = abs(tx_amount) if tx_amount is not None else None

            # ------------------------------------------------------------------
            # Variance reasoning + suggested_adjustment (Exit Criteria 1 & 2)
            # ------------------------------------------------------------------
            statement_desc = stmt.get("raw_desc", "") or ""
            ledger_descs = _collect_ledger_descs_for_line(match_info, ledger_rows)

            try:
                variance_meta = compute_variance_metadata(
                    tx_amount=tx_amount,
                    variance_amount=variance_amount,
                    currency=currency,
                    statement_desc=statement_desc,
                    ledger_descs=ledger_descs,
                    tol_abs=tol_abs,
                    tol_pct=tol_pct,
                )
            except Exception as e:  # pragma: no cover
                log.warning(
                    "compute_variance_metadata failed for stmt_row_id=%s: %s",
                    stmt_row_id,
                    e,
                )
                variance_meta = {
                    "variance_reason": None,
                    "auto_apply_safe": False,
                    "diagnostic": {},
                    "suggested_adjustment": None,
                }

            # Build raw payload
            payload: Dict[str, Any] = {
                "engine": "bank_reco_v1",
                "org_id": org_id,
                "user_id": user_id,
                "run_meta": run_meta_safe,
                "statement_line_index": line_index,
                "statement_row_id": stmt_row_id,
                "statement": stmt,
                "matches_for_line": match_info or {},
                "recommendation_for_line": reco_info or {},
                "ledger_snapshot": {
                    "size": len(ledger_rows),
                },
                "summary_snapshot": matcher_result.get("summary", {}),
                # New variance meta for UI/debug:
                "variance_reason": variance_meta.get("variance_reason"),
                "auto_apply_safe": variance_meta.get("auto_apply_safe"),
                "variance_meta": variance_meta.get("diagnostic"),
            }

            # Make the payload JSON-safe (fixes Timestamp / datetime issues)
            safe_payload = _to_json_safe(payload)

            # Make suggested_adjustment JSON-safe as well
            safe_suggested_adj = _to_json_safe(
                variance_meta.get("suggested_adjustment")
            )

            recon_item = ReconItem(
                tx_id=tx_id_val,
                source_system=source_system,
                employee_id=raw.get("employee_id") or raw.get("Employee ID"),
                currency=currency,
                tx_amount=tx_amount,
                receipt_amount=None,  # bank-vs-ledger only
                match_status=match_status,
                variance_type=variance_type,
                variance_amount=variance_amount,
                variance_payload=safe_payload,
                suggested_adjustment=safe_suggested_adj,
                owner_id=user_id,
            )

            db.add(recon_item)
            created_items.append(recon_item)

            # ✅ Initialize SLA fields using existing helper
            apply_default_sla(recon_item, now)

        # Flush to get primary keys if needed
        await db.flush()
        # Commit transaction
        await db.commit()

        # Refresh instances to populate any DB-side defaults
        for item in created_items:
            await db.refresh(item)

        return created_items

    except SQLAlchemyError:
        await db.rollback()
        raise
