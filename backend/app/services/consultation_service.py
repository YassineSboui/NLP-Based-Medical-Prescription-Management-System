"""Storing, querying, exporting and auditing consultations.

All database access for consultations goes through here, so the routes stay
thin and the audit record cannot be skipped by a caller who forgets: writing a
consultation and writing its audit event happen in the same transaction.
"""

from __future__ import annotations

import csv
import io
import json
import uuid
from datetime import datetime, timezone

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.models import AuditEvent as AuditEventRow
from app.db.models import Consultation as ConsultationRow
from app.models.schemas import (
    AuditEvent,
    ConsultationDetail,
    ConsultationSummary,
    EngineMetrics,
    EngineOpinion,
    ExtractedEntities,
    HistoryStatsResponse,
    MedicationRecommendation,
)

EXPORT_COLUMNS = [
    "consultation_id",
    "session_id",
    "batch_id",
    "created_at",
    "patient_reference",
    "source",
    "original_text",
    "cleaned_text",
    "predicted_disease",
    "confidence",
    "abstained",
    "decided_by_engine",
    "engine_name",
    "engine_accuracy",
    "engine_metrics_basis",
    "policy",
    "policy_reason",
    "extracted_symptoms",
    "recommended_medications",
]


def new_id() -> str:
    return str(uuid.uuid4())


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ConsultationService:
    """Persistence and retrieval for consultation records."""

    # -- writing -----------------------------------------------------------

    def record(
        self,
        session: Session,
        *,
        consultation_id: str,
        session_id: str,
        batch_id: str | None,
        patient_reference: str | None,
        source: str,
        original_text: str,
        cleaned_text: str,
        requested_model_key: str,
        decision,
        entities: ExtractedEntities,
        opinions: list[EngineOpinion],
        actions: list[str],
        medications: list[MedicationRecommendation],
        disclaimer: str,
    ) -> ConsultationRow:
        """Write a consultation and its audit event in one transaction."""
        engine = decision.decided_by
        created_at = utc_now()

        row = ConsultationRow(
            id=consultation_id,
            session_id=session_id,
            batch_id=batch_id,
            created_at=created_at,
            patient_reference=patient_reference,
            source=source,
            original_text=original_text,
            cleaned_text=cleaned_text,
            requested_model_key=requested_model_key,
            engine_key=engine.key,
            engine_name=engine.name,
            policy=decision.policy,
            policy_reason=decision.policy_reason,
            predicted_disease=decision.predicted_disease,
            confidence=decision.confidence,
            abstained=decision.abstained,
            entities_json=entities.model_dump_json(),
            opinions_json=json.dumps([opinion.model_dump() for opinion in opinions]),
            actions_json=json.dumps(actions),
            medications_json=json.dumps([item.model_dump() for item in medications]),
            engine_metrics_json=engine.metrics.model_dump_json(),
            disclaimer=disclaimer,
        )
        session.add(row)

        session.add(
            AuditEventRow(
                id=new_id(),
                consultation_id=consultation_id,
                created_at=created_at,
                event_type="recommendation_issued",
                engine_key=engine.key,
                engine_name=engine.name,
                # Snapshot, not a lookup: retraining moves these numbers and the
                # audit trail has to preserve what was claimed at the time.
                engine_accuracy=engine.metrics.accuracy,
                engine_metrics_basis=engine.metrics.basis,
                predicted_disease=decision.predicted_disease,
                confidence=decision.confidence,
                suggested_medications_json=json.dumps([item.name for item in medications]),
                detail=decision.policy_reason,
                sequence=1,
            )
        )
        return row

    # -- reading -----------------------------------------------------------

    @staticmethod
    def _base_query(
        session_id: str | None,
        disease: str | None,
        engine_key: str | None,
        since: datetime | None,
        until: datetime | None,
    ):
        statement = select(ConsultationRow)
        if session_id:
            statement = statement.where(ConsultationRow.session_id == session_id)
        if disease:
            statement = statement.where(ConsultationRow.predicted_disease == disease)
        if engine_key:
            statement = statement.where(ConsultationRow.engine_key == engine_key)
        if since:
            statement = statement.where(ConsultationRow.created_at >= since)
        if until:
            statement = statement.where(ConsultationRow.created_at <= until)
        return statement

    def list(
        self,
        session: Session,
        *,
        limit: int = 50,
        offset: int = 0,
        session_id: str | None = None,
        disease: str | None = None,
        engine_key: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> tuple[int, list[ConsultationRow]]:
        statement = self._base_query(session_id, disease, engine_key, since, until)
        total = session.scalar(select(func.count()).select_from(statement.subquery())) or 0
        rows = session.scalars(
            statement.order_by(ConsultationRow.created_at.desc(), ConsultationRow.id.desc())
            .limit(limit)
            .offset(offset)
        ).all()
        return int(total), list(rows)

    def get(self, session: Session, consultation_id: str) -> ConsultationRow | None:
        return session.get(ConsultationRow, consultation_id)

    def audit_events(self, session: Session, consultation_id: str) -> list[AuditEventRow]:
        return list(
            session.scalars(
                select(AuditEventRow)
                .where(AuditEventRow.consultation_id == consultation_id)
                .order_by(AuditEventRow.created_at, AuditEventRow.sequence)
            ).all()
        )

    def stats(self, session: Session) -> HistoryStatsResponse:
        total = session.scalar(select(func.count()).select_from(ConsultationRow)) or 0
        sessions = session.scalar(select(func.count(func.distinct(ConsultationRow.session_id)))) or 0
        abstentions = (
            session.scalar(
                select(func.count()).select_from(ConsultationRow).where(ConsultationRow.abstained.is_(True))
            )
            or 0
        )
        by_disease = dict(
            session.execute(
                select(ConsultationRow.predicted_disease, func.count()).group_by(ConsultationRow.predicted_disease)
            ).all()
        )
        by_engine = dict(
            session.execute(
                select(ConsultationRow.engine_key, func.count()).group_by(ConsultationRow.engine_key)
            ).all()
        )
        first = session.scalar(select(func.min(ConsultationRow.created_at)))
        last = session.scalar(select(func.max(ConsultationRow.created_at)))

        return HistoryStatsResponse(
            total_consultations=int(total),
            total_sessions=int(sessions),
            abstentions=int(abstentions),
            by_disease={str(key): int(value) for key, value in by_disease.items()},
            by_engine={str(key): int(value) for key, value in by_engine.items()},
            first_consultation_at=first,
            last_consultation_at=last,
        )

    # -- conversion --------------------------------------------------------

    @staticmethod
    def to_summary(row: ConsultationRow) -> ConsultationSummary:
        return ConsultationSummary(
            consultation_id=row.id,
            session_id=row.session_id,
            created_at=row.created_at,
            patient_reference=row.patient_reference,
            source=row.source,
            batch_id=row.batch_id,
            original_text=row.original_text,
            predicted_disease=row.predicted_disease,
            confidence=row.confidence,
            engine_key=row.engine_key,
            engine_name=row.engine_name,
            abstained=row.abstained,
        )

    @classmethod
    def to_detail(cls, row: ConsultationRow) -> ConsultationDetail:
        return ConsultationDetail(
            **cls.to_summary(row).model_dump(),
            cleaned_text=row.cleaned_text,
            requested_model_key=row.requested_model_key,
            policy=row.policy,
            policy_reason=row.policy_reason,
            engine_metrics=EngineMetrics.model_validate_json(row.engine_metrics_json),
            extracted_entities=ExtractedEntities.model_validate_json(row.entities_json),
            engine_opinions=[EngineOpinion(**item) for item in json.loads(row.opinions_json)],
            recommended_actions=json.loads(row.actions_json),
            recommended_medicines=[MedicationRecommendation(**item) for item in json.loads(row.medications_json)],
            disclaimer=row.disclaimer,
        )

    @staticmethod
    def to_audit_event(row: AuditEventRow) -> AuditEvent:
        return AuditEvent(
            audit_id=row.id,
            consultation_id=row.consultation_id,
            created_at=row.created_at,
            event_type=row.event_type,
            engine_key=row.engine_key,
            engine_name=row.engine_name,
            engine_accuracy=row.engine_accuracy,
            engine_metrics_basis=row.engine_metrics_basis,
            predicted_disease=row.predicted_disease,
            confidence=row.confidence,
            suggested_medications=json.loads(row.suggested_medications_json),
            detail=row.detail,
        )

    # -- export ------------------------------------------------------------

    @classmethod
    def to_csv(cls, rows: list[ConsultationRow]) -> str:
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=EXPORT_COLUMNS, lineterminator="\n")
        writer.writeheader()

        for row in rows:
            entities = json.loads(row.entities_json)
            medications = json.loads(row.medications_json)
            metrics = json.loads(row.engine_metrics_json)
            writer.writerow(
                {
                    "consultation_id": row.id,
                    "session_id": row.session_id,
                    "batch_id": row.batch_id or "",
                    "created_at": row.created_at.isoformat(),
                    "patient_reference": row.patient_reference or "",
                    "source": row.source,
                    "original_text": row.original_text,
                    "cleaned_text": row.cleaned_text,
                    "predicted_disease": row.predicted_disease,
                    "confidence": row.confidence,
                    "abstained": row.abstained,
                    "decided_by_engine": row.engine_key,
                    "engine_name": row.engine_name,
                    "engine_accuracy": metrics.get("accuracy", ""),
                    "engine_metrics_basis": metrics.get("basis", ""),
                    "policy": row.policy,
                    "policy_reason": row.policy_reason,
                    "extracted_symptoms": "; ".join(entities.get("symptoms", [])),
                    "recommended_medications": "; ".join(item["name"] for item in medications),
                }
            )

        return buffer.getvalue()

    @classmethod
    def to_report(cls, row: ConsultationRow) -> str:
        """A consultation as a plain-text/Markdown report.

        Deliberately leads with the attribution and the caveats rather than the
        prediction, so a printed copy cannot be mistaken for a diagnosis.
        """
        detail = cls.to_detail(row)
        metrics = detail.engine_metrics
        lines = [
            "# Symptom analysis report",
            "",
            f"- Consultation: `{detail.consultation_id}`",
            f"- Session: `{detail.session_id}`",
            f"- Recorded: {detail.created_at.isoformat()}",
        ]
        if detail.patient_reference:
            lines.append(f"- Case reference: {detail.patient_reference}")

        lines += [
            "",
            "## Not a diagnosis",
            "",
            detail.disclaimer,
            "",
            "## Which engine answered",
            "",
            f"- Engine: **{detail.engine_name}** (`{detail.engine_key}`)",
            f"- Arbitration policy: `{detail.policy}`",
            f"- Why this engine: {detail.policy_reason}",
            f"- Engine accuracy: {metrics.accuracy if metrics.accuracy is not None else 'not evaluated'}"
            + (f" (95% CI {metrics.accuracy_ci_95})" if metrics.accuracy_ci_95 else ""),
            f"- Measured on: {metrics.basis}",
        ]
        if metrics.cross_source_transfer_recall is not None:
            lines.append(
                f"- Cross-source transfer recall: {metrics.cross_source_transfer_recall} "
                "(how it does on wording from a source it never trained on)"
            )

        lines += [
            "",
            "## Input",
            "",
            f"> {detail.original_text}",
            "",
            "## Result",
            "",
            f"- Suggested condition: **{detail.predicted_disease}**",
            f"- Confidence: {detail.confidence}",
            f"- Abstained: {'yes' if detail.abstained else 'no'}",
            "",
            "### Extracted entities",
            "",
            f"- Symptoms: {', '.join(detail.extracted_entities.symptoms) or 'none'}",
            f"- Diseases mentioned: {', '.join(detail.extracted_entities.diseases) or 'none'}",
            f"- Medications mentioned: {', '.join(detail.extracted_entities.medications) or 'none'}",
            f"- Dosage mentions: {', '.join(detail.extracted_entities.dosage_mentions) or 'none'}",
            "",
            "### Every engine's opinion",
            "",
        ]
        for opinion in detail.engine_opinions:
            marker = " <- decided" if opinion.engine == detail.engine_key else ""
            lines.append(
                f"- `{opinion.engine}`: {opinion.disease or 'no opinion'} "
                f"({opinion.confidence if opinion.confidence is not None else 'n/a'}){marker}"
            )
            lines.append(f"  - {opinion.detail}")

        lines += ["", "### Recommended actions", ""]
        lines += [f"- {action}" for action in detail.recommended_actions] or ["- none recorded"]

        lines += ["", "### Medication information", ""]
        if not detail.recommended_medicines:
            lines.append("- none recorded")
        for medication in detail.recommended_medicines:
            lines.append(f"- **{medication.name}**")
            lines.append(f"  - Dosage: {medication.standard_dosage}")
            lines.append(f"  - Administration: {medication.administration}")
            for warning in medication.warnings:
                lines.append(f"  - Warning: {warning}")

        lines += ["", "## Audit trail", ""]
        for event in row.audit_events:
            lines.append(
                f"- {event.created_at.isoformat()} | {event.event_type} | engine=`{event.engine_key}` "
                f"| accuracy={event.engine_accuracy} | suggested="
                f"{', '.join(json.loads(event.suggested_medications_json)) or 'none'}"
            )

        return "\n".join(lines) + "\n"
