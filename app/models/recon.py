from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Column,
    Integer,
    String,
    DateTime,
    Numeric,
    Boolean,
    JSON,
    ForeignKey,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship

from app.db.base import Base


class ReconItem(Base):
    """
    One reconciliation item (case) that appears in the Recon Insight workspace.

    Comes from UC1 world (tx + bank/card/GL lines) and is enriched with
    variance analysis, suggested adjustments, and SLA/owner info.
    """

    __tablename__ = "recon_items"
    __table_args__ = {"schema": "erp_ai"}  # IMPORTANT: match your schema

    id = Column(Integer, primary_key=True, index=True)

    # Foreign reference into UC1 world
    tx_id = Column(Integer, index=True, nullable=False)
    source_system = Column(String(32), nullable=True)  # e.g. 'CARD', 'BANK'

    employee_id = Column(String(64), index=True, nullable=True)
    currency = Column(String(8), nullable=False)
    tx_amount = Column(Numeric(18, 2), nullable=False)
    receipt_amount = Column(Numeric(18, 2), nullable=True)

    # Recon status + variance
    match_status = Column(
        String(32),
        index=True,
        nullable=False,
        default="unmatched",  # unmatched / matched / ignored / pending
    )
    variance_type = Column(String(64), index=True, nullable=True)
    variance_amount = Column(Numeric(18, 2), nullable=True)
    variance_payload = Column(JSON, nullable=True)

    # Suggested journal entry (produced by engine)
    # shape:
    # {
    #   "lines": [...],
    #   "notes": "Variance not within auto-apply tolerance.",
    #   "amount": 1780.49,
    #   "reason": "fx_variance" | "tip" | "partial" | "other",
    #   "currency": "AED",
    #   "auto_apply_safe": false
    # }
    suggested_adjustment = Column(JSON, nullable=True)

    # Ownership & SLA
    owner_id = Column(String(64), index=True, nullable=True)
    opened_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    closed_at = Column(DateTime, nullable=True)

    sla_hours = Column(Integer, nullable=True)
    sla_due_at = Column(DateTime, nullable=True)
    sla_breached = Column(Boolean, nullable=False, default=False)

    # Resolution summary fields (latest status)
    # We still keep these for quick reporting, even though the
    # detailed history lives in ReconAction.
    # expected values: "applied", "ignored", "reopened", etc.
    resolution_status = Column(String(32), nullable=True)
    resolution_note = Column(String, nullable=True)
    resolved_by = Column(String(64), nullable=True)
    resolved_at = Column(DateTime, nullable=True)

    # Audit fields
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    # Full action history for this item (apply/ignore/system notes)
    actions = relationship(
        "ReconAction",
        back_populates="recon_item",
        cascade="all, delete-orphan",
        lazy="selectin",
    )


class ReconAction(Base):
    """
    Immutable audit log of actions taken on a ReconItem.

    This follows the older ReconCaseAction field naming:
    - performed_by, performed_at, notes, payload (JSONB)

    Examples:
    - "apply"   (user clicked Apply suggested adjustment)
    - "ignore"  (user chose to ignore this variance)
    - "note"    (manual comment / note)
    - "system"  (auto-open, SLA breach, auto-close, etc.)
    """

    __tablename__ = "recon_actions"
    __table_args__ = {"schema": "erp_ai"}

    id = Column(Integer, primary_key=True, index=True)
    recon_item_id = Column(
        Integer,
        ForeignKey("erp_ai.recon_items.id"),
        nullable=False,
        index=True,
    )

    # "apply", "ignore", "reopen", "update_owner", "sla_breach", etc.
    action_type = Column(String(50), nullable=False)

    # who performed it (AAD UPN / user_id / email / service principal id)
    performed_by = Column(String(255), nullable=False)

    performed_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # Optional human-readable note
    notes = Column(Text, nullable=True)

    # free-form payload: previous status, suggested_adjustment, posting result, etc.
    payload = Column(JSONB, nullable=True)

    # Optional: freeze suggested_adjustment at time of APPLY (for audit)
    adjustment_snapshot = Column(JSONB, nullable=True)

    recon_item = relationship("ReconItem", back_populates="actions")


class BankRecoRun(Base):
    """
    Stores the full JSON output of the bank reconciliation matcher
    for a given run (statement file + ledger file + params).
    """

    __tablename__ = "bank_reco_runs"
    __table_args__ = {"schema": "erp_ai"}

    id = Column(Integer, primary_key=True, index=True)

    org_id = Column(String(64), index=True, nullable=True)
    user_id = Column(String(64), index=True, nullable=True)

    statement_filename = Column(String(256), nullable=False)
    ledger_filename = Column(String(256), nullable=False)

    params = Column(JSON, nullable=False, default=dict)
    summary = Column(JSON, nullable=False, default=dict)
    result = Column(JSON, nullable=False, default=dict)

    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )
