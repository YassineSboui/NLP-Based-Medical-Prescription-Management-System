"""SQLite tables for consultations and the audit trail.

A tool that suggests medications and keeps no record of what it suggested
cannot be reviewed after the fact, which is the whole point of an audit trail.
Two tables, both append-only in practice:

``consultations``
    One row per analysed note. Keeps the input, the cleaned text, the decision,
    which engine made it and why, and the full recommendation payload, so a
    record can be reproduced exactly as the user saw it.

``audit_events``
    What was suggested, when, and by which engine -- including a *snapshot of
    that engine's accuracy at the time*. Retraining changes an engine's metrics,
    so without the snapshot you could never reconstruct what the system was
    claiming about itself on the day it made a given suggestion.

SQLite via SQLAlchemy keeps the whole thing a single file with no service to
run, which preserves the project's best property: it works offline with nothing
to sign up for.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Consultation(Base):
    __tablename__ = "consultations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    session_id: Mapped[str] = mapped_column(String(36), index=True)
    batch_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, index=True)

    patient_reference: Mapped[str | None] = mapped_column(String(128), nullable=True)
    source: Mapped[str] = mapped_column(String(16), default="api")

    original_text: Mapped[str] = mapped_column(Text)
    cleaned_text: Mapped[str] = mapped_column(Text)

    requested_model_key: Mapped[str] = mapped_column(String(32))
    engine_key: Mapped[str] = mapped_column(String(32), index=True)
    engine_name: Mapped[str] = mapped_column(String(128))
    policy: Mapped[str] = mapped_column(String(64))
    policy_reason: Mapped[str] = mapped_column(Text)

    predicted_disease: Mapped[str] = mapped_column(String(64), index=True)
    confidence: Mapped[float] = mapped_column(Float)
    abstained: Mapped[bool] = mapped_column(Boolean, default=False)

    # JSON-encoded payloads. Stored verbatim so a record can be replayed exactly
    # as it was returned, even after the schemas or the knowledge base change.
    entities_json: Mapped[str] = mapped_column(Text)
    opinions_json: Mapped[str] = mapped_column(Text)
    actions_json: Mapped[str] = mapped_column(Text)
    medications_json: Mapped[str] = mapped_column(Text)
    engine_metrics_json: Mapped[str] = mapped_column(Text)
    disclaimer: Mapped[str] = mapped_column(Text)

    audit_events: Mapped[list["AuditEvent"]] = relationship(
        back_populates="consultation", cascade="all, delete-orphan", order_by="AuditEvent.created_at"
    )


class AuditEvent(Base):
    __tablename__ = "audit_events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    consultation_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("consultations.id", ondelete="CASCADE"), index=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, index=True)

    event_type: Mapped[str] = mapped_column(String(32))
    engine_key: Mapped[str] = mapped_column(String(32))
    engine_name: Mapped[str] = mapped_column(String(128))

    # Snapshotted on purpose: retraining moves an engine's metrics, and the
    # audit record has to show what was claimed at the time of the suggestion.
    engine_accuracy: Mapped[float | None] = mapped_column(Float, nullable=True)
    engine_metrics_basis: Mapped[str | None] = mapped_column(Text, nullable=True)

    predicted_disease: Mapped[str] = mapped_column(String(64))
    confidence: Mapped[float] = mapped_column(Float)
    suggested_medications_json: Mapped[str] = mapped_column(Text)
    detail: Mapped[str] = mapped_column(Text)

    sequence: Mapped[int] = mapped_column(Integer, default=0)

    consultation: Mapped[Consultation] = relationship(back_populates="audit_events")


Index("ix_consultations_session_created", Consultation.session_id, Consultation.created_at)
