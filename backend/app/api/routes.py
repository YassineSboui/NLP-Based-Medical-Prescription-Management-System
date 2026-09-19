"""HTTP surface.

Routes stay thin: they validate input, call a service, and shape the response.
Persisting a consultation and writing its audit event happen together inside
``ConsultationService.record``, so no route can accidentally produce a
medication suggestion that leaves no record.
"""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy import text as sql_text
from sqlalchemy.orm import Session

from app.api.dependencies import require_api_key
from app.core.config import get_settings
from app.db.session import get_session
from app.models.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    AuditListResponse,
    BatchAnalyzeRequest,
    BatchAnalyzeResponse,
    BatchItemResult,
    ConsultationDetail,
    ConsultationListResponse,
    ConsultationReport,
    EngineListResponse,
    HealthResponse,
    HistoryStatsResponse,
)
from app.services.consultation_service import ConsultationService, new_id, utc_now
from app.services.disease_prediction_service import (
    ENSEMBLE_POLICY,
    MODEL_CONFIDENCE_FLOOR,
    DiseasePredictionService,
)
from app.services.nlp_service import Analysis, NLPService

API_VERSION = "2.0.0"

POLICY_DESCRIPTION = (
    "The requested statistical engine answers when its probability is at or above "
    "model_confidence_floor. Below that it is treated as not knowing, and the "
    "symptom-profile rule engine answers instead. If neither has an answer the "
    "ensemble abstains and returns 'unknown'. decision.decided_by always names the "
    "engine that actually answered and carries that engine's own metrics."
)

router = APIRouter()
nlp_service = NLPService()
consultations = ConsultationService()


def _prediction_service() -> DiseasePredictionService:
    return nlp_service.predictor


def _to_response(
    analysis: Analysis,
    *,
    consultation_id: str | None,
    session_id: str,
    created_at: datetime,
) -> AnalyzeResponse:
    decision = analysis.to_decision_schema()
    return AnalyzeResponse(
        consultation_id=consultation_id,
        session_id=session_id,
        created_at=created_at,
        original_text=analysis.original_text,
        cleaned_text=analysis.cleaned_text,
        extracted_entities=analysis.entities,
        predicted_disease=decision.predicted_disease,
        confidence=decision.confidence,
        decision=decision,
        engine_opinions=analysis.decision.opinions,
        # Alias kept for existing clients. It now points at whichever engine
        # actually decided, which is the bug this rework set out to fix.
        model_used=decision.decided_by,
        recommended_actions=analysis.recommended_actions,
        recommended_medicines=analysis.recommended_medicines,
        disclaimer=analysis.disclaimer,
    )


def _validate_text(text: str) -> str:
    settings = get_settings()
    if len(text) > settings.max_text_length:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Text exceeds the {settings.max_text_length} character limit.",
        )
    return text


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

@router.get("/health", response_model=HealthResponse, tags=["service"])
def health_check(session: Session = Depends(get_session)) -> HealthResponse:
    settings = get_settings()
    try:
        session.execute(sql_text("SELECT 1"))
        database_ready = True
    except Exception:  # pragma: no cover - only on a broken database
        database_ready = False

    predictor = _prediction_service()
    return HealthResponse(
        status="ok",
        service="medical-prescription-nlp",
        version=API_VERSION,
        auth_required=settings.auth_required,
        database_ready=database_ready,
        engines_available={key: engine.is_available() for key, engine in predictor.engines.items()},
    )


@router.get("/engines", response_model=EngineListResponse, tags=["engines"])
def list_engines() -> EngineListResponse:
    """Every engine, with the metrics that belong to it."""
    predictor = _prediction_service()
    return EngineListResponse(
        default_engine="classical",
        ensemble_policy=ENSEMBLE_POLICY,
        policy_description=POLICY_DESCRIPTION,
        model_confidence_floor=MODEL_CONFIDENCE_FLOOR,
        engines=predictor.list_engines(),
    )


@router.get(
    "/models",
    response_model=EngineListResponse,
    tags=["engines"],
    deprecated=True,
    summary="Deprecated alias of /engines",
)
def list_models() -> EngineListResponse:
    """Kept so existing clients keep working. Use /engines."""
    return list_engines()


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    tags=["analysis"],
    dependencies=[Depends(require_api_key)],
)
def analyze_symptoms(request: AnalyzeRequest, session: Session = Depends(get_session)) -> AnalyzeResponse:
    _validate_text(request.text)
    session_id = request.session_id or new_id()
    analysis = nlp_service.analyze(request.text, model_key=request.model_key)

    if not request.persist:
        return _to_response(analysis, consultation_id=None, session_id=session_id, created_at=utc_now())

    consultation_id = new_id()
    row = consultations.record(
        session,
        consultation_id=consultation_id,
        session_id=session_id,
        batch_id=None,
        patient_reference=request.patient_reference,
        source="api",
        original_text=analysis.original_text,
        cleaned_text=analysis.cleaned_text,
        requested_model_key=request.model_key,
        decision=analysis.to_decision_schema(),
        entities=analysis.entities,
        opinions=analysis.decision.opinions,
        actions=analysis.recommended_actions,
        medications=analysis.recommended_medicines,
        disclaimer=analysis.disclaimer,
    )
    return _to_response(
        analysis, consultation_id=consultation_id, session_id=session_id, created_at=row.created_at
    )


@router.post(
    "/analyze/batch",
    response_model=BatchAnalyzeResponse,
    tags=["analysis"],
    dependencies=[Depends(require_api_key)],
)
def analyze_batch(request: BatchAnalyzeRequest, session: Session = Depends(get_session)) -> BatchAnalyzeResponse:
    """Analyse several notes in one call.

    One bad note does not fail the batch: each item reports its own ok/error so
    a caller processing a day's worth of notes gets partial results rather than
    a single 422 and nothing to show for it.
    """
    settings = get_settings()
    if len(request.notes) > settings.max_batch_items:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"A batch may contain at most {settings.max_batch_items} notes.",
        )

    session_id = request.session_id or new_id()
    batch_id = new_id()
    created_at = utc_now()
    results: list[BatchItemResult] = []

    for index, note in enumerate(request.notes):
        try:
            if not note or len(note.strip()) < 3:
                raise ValueError("Note must contain at least 3 characters.")
            _validate_text(note)

            analysis = nlp_service.analyze(note, model_key=request.model_key)
            consultation_id = None
            item_created_at = created_at

            if request.persist:
                consultation_id = new_id()
                row = consultations.record(
                    session,
                    consultation_id=consultation_id,
                    session_id=session_id,
                    batch_id=batch_id,
                    patient_reference=None,
                    source="batch",
                    original_text=analysis.original_text,
                    cleaned_text=analysis.cleaned_text,
                    requested_model_key=request.model_key,
                    decision=analysis.to_decision_schema(),
                    entities=analysis.entities,
                    opinions=analysis.decision.opinions,
                    actions=analysis.recommended_actions,
                    medications=analysis.recommended_medicines,
                    disclaimer=analysis.disclaimer,
                )
                item_created_at = row.created_at

            results.append(
                BatchItemResult(
                    index=index,
                    ok=True,
                    result=_to_response(
                        analysis,
                        consultation_id=consultation_id,
                        session_id=session_id,
                        created_at=item_created_at,
                    ),
                )
            )
        except HTTPException as exc:
            results.append(BatchItemResult(index=index, ok=False, error=str(exc.detail)))
        except Exception as exc:  # noqa: BLE001 - one bad note must not fail the batch
            results.append(BatchItemResult(index=index, ok=False, error=str(exc)))

    succeeded = sum(1 for item in results if item.ok)
    return BatchAnalyzeResponse(
        batch_id=batch_id,
        session_id=session_id,
        created_at=created_at,
        submitted=len(request.notes),
        succeeded=succeeded,
        failed=len(results) - succeeded,
        results=results,
    )


# ---------------------------------------------------------------------------
# History, audit, export
# ---------------------------------------------------------------------------

@router.get(
    "/consultations",
    response_model=ConsultationListResponse,
    tags=["history"],
    dependencies=[Depends(require_api_key)],
)
def list_consultations(
    session: Session = Depends(get_session),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    session_id: str | None = None,
    disease: str | None = None,
    engine_key: str | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
) -> ConsultationListResponse:
    total, rows = consultations.list(
        session,
        limit=limit,
        offset=offset,
        session_id=session_id,
        disease=disease,
        engine_key=engine_key,
        since=since,
        until=until,
    )
    return ConsultationListResponse(
        total=total,
        limit=limit,
        offset=offset,
        items=[consultations.to_summary(row) for row in rows],
    )


@router.get(
    "/consultations/stats",
    response_model=HistoryStatsResponse,
    tags=["history"],
    dependencies=[Depends(require_api_key)],
)
def consultation_stats(session: Session = Depends(get_session)) -> HistoryStatsResponse:
    return consultations.stats(session)


@router.get(
    "/consultations/export",
    tags=["history"],
    dependencies=[Depends(require_api_key)],
    response_class=Response,
    summary="Export filtered consultations as CSV",
)
def export_consultations(
    session: Session = Depends(get_session),
    limit: int = Query(1000, ge=1, le=10000),
    session_id: str | None = None,
    disease: str | None = None,
    engine_key: str | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
) -> Response:
    _, rows = consultations.list(
        session,
        limit=limit,
        offset=0,
        session_id=session_id,
        disease=disease,
        engine_key=engine_key,
        since=since,
        until=until,
    )
    filename = f"consultations-{utc_now().strftime('%Y%m%d-%H%M%S')}.csv"
    return Response(
        content=consultations.to_csv(rows),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get(
    "/consultations/{consultation_id}",
    response_model=ConsultationDetail,
    tags=["history"],
    dependencies=[Depends(require_api_key)],
)
def get_consultation(consultation_id: str, session: Session = Depends(get_session)) -> ConsultationDetail:
    row = consultations.get(session, consultation_id)
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Consultation not found.")
    return consultations.to_detail(row)


@router.get(
    "/consultations/{consultation_id}/audit",
    response_model=AuditListResponse,
    tags=["history"],
    dependencies=[Depends(require_api_key)],
)
def get_audit_trail(consultation_id: str, session: Session = Depends(get_session)) -> AuditListResponse:
    if consultations.get(session, consultation_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Consultation not found.")
    events = consultations.audit_events(session, consultation_id)
    return AuditListResponse(
        consultation_id=consultation_id,
        events=[consultations.to_audit_event(event) for event in events],
    )


@router.get(
    "/consultations/{consultation_id}/report",
    response_model=ConsultationReport,
    tags=["history"],
    dependencies=[Depends(require_api_key)],
)
def get_consultation_report(
    consultation_id: str, session: Session = Depends(get_session)
) -> ConsultationReport:
    row = consultations.get(session, consultation_id)
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Consultation not found.")
    return ConsultationReport(
        consultation_id=consultation_id,
        generated_at=utc_now(),
        format="markdown",
        content=consultations.to_report(row),
    )
