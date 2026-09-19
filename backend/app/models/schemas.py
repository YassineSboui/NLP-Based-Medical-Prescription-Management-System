"""Request and response contracts for the API.

The important change here is that a prediction now carries its own attribution.
``AnalyzeResponse.decision`` names the engine that produced the answer and
carries that engine's own measured metrics, and ``engine_opinions`` shows what
every engine thought, including the ones that were overruled. A client asking
"which one answered?" gets a true answer.

``model_used`` is kept as an alias of ``decision.decided_by`` so existing
clients do not break. Unlike before, it now points at whichever engine actually
decided.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

# Pydantic reserves the `model_` prefix. These fields are part of the published
# contract and predate that reservation, so the namespace guard is turned off
# rather than renaming fields clients already depend on.
_ALLOW_MODEL_PREFIX = ConfigDict(protected_namespaces=())


# ---------------------------------------------------------------------------
# Engines
# ---------------------------------------------------------------------------

class EngineMetrics(BaseModel):
    """How well one engine actually does, measured on a shared held-out split."""

    accuracy: float | None = Field(None, description="Accuracy on the held-out test set.")
    accuracy_ci_95: list[float] | None = Field(None, description="95% Wilson interval for accuracy.")
    macro_f1: float | None = None
    external_use_case_accuracy: float | None = Field(
        None, description="Accuracy on the hand-written scenarios that are not in the dataset."
    )
    cross_source_transfer_recall: float | None = Field(
        None,
        description=(
            "Recall when an entire source passage is held out of training. The honest "
            "measure of generalisation, and much lower than the headline accuracy."
        ),
    )
    basis: str = Field(..., description="What these numbers were measured on.")
    evaluated_at: str | None = None
    source_file: str | None = None


class EngineInfo(BaseModel):
    model_config = _ALLOW_MODEL_PREFIX

    key: str
    name: str
    family: str
    description: str
    confidence_meaning: str = Field(
        ..., description="What this engine's confidence number actually means."
    )
    is_available: bool = True
    is_default: bool = False
    accuracy: float | None = Field(None, description="Convenience mirror of metrics.accuracy.")
    macro_f1: float | None = Field(None, description="Convenience mirror of metrics.macro_f1.")
    metrics: EngineMetrics


class EngineOpinion(BaseModel):
    """One engine's candidate answer, whether or not it was used."""

    engine: str
    disease: str | None = None
    confidence: float | None = None
    available: bool = True
    detail: str
    matched_symptoms: list[str] = Field(default_factory=list)


class EngineListResponse(BaseModel):
    model_config = _ALLOW_MODEL_PREFIX

    default_engine: str
    ensemble_policy: str
    policy_description: str
    model_confidence_floor: float
    engines: list[EngineInfo]


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

class ExtractedEntities(BaseModel):
    symptoms: list[str]
    diseases: list[str]
    medications: list[str]
    dosage_mentions: list[str]


class MedicationRecommendation(BaseModel):
    name: str
    standard_dosage: str
    administration: str
    warnings: list[str]


class Decision(BaseModel):
    """Who decided, what they decided, and why they were the one to decide."""

    model_config = _ALLOW_MODEL_PREFIX

    predicted_disease: str
    confidence: float
    decided_by: EngineInfo
    policy: str
    policy_reason: str
    abstained: bool = Field(
        False, description="True when no engine was confident enough and the label is 'unknown'."
    )


class AnalyzeRequest(BaseModel):
    model_config = _ALLOW_MODEL_PREFIX

    text: str = Field(..., min_length=3, description="Patient-reported symptoms in natural language.")
    model_key: str = Field(
        "classical", description="Preferred statistical engine: 'classical' or 'advanced'."
    )
    session_id: str | None = Field(
        None, description="Groups consultations. Generated and returned if omitted."
    )
    patient_reference: str | None = Field(
        None,
        max_length=128,
        description=(
            "Optional free-text label for the case, e.g. a local case number. Stored as given; "
            "do not put identifying patient data in it."
        ),
    )
    persist: bool = Field(True, description="Set false to analyse without writing a consultation record.")


class AnalyzeResponse(BaseModel):
    model_config = _ALLOW_MODEL_PREFIX

    consultation_id: str | None = Field(None, description="Null when persist=false.")
    session_id: str
    created_at: datetime
    original_text: str
    cleaned_text: str
    extracted_entities: ExtractedEntities
    predicted_disease: str
    confidence: float
    decision: Decision
    engine_opinions: list[EngineOpinion]
    model_used: EngineInfo = Field(
        ..., description="Alias of decision.decided_by: the engine that actually answered."
    )
    recommended_actions: list[str]
    recommended_medicines: list[MedicationRecommendation]
    disclaimer: str


class BatchAnalyzeRequest(BaseModel):
    model_config = _ALLOW_MODEL_PREFIX

    notes: list[str] = Field(..., min_length=1, description="One free-text note per item.")
    model_key: str = "classical"
    session_id: str | None = None
    persist: bool = True


class BatchItemResult(BaseModel):
    index: int
    ok: bool
    error: str | None = None
    result: AnalyzeResponse | None = None


class BatchAnalyzeResponse(BaseModel):
    batch_id: str
    session_id: str
    created_at: datetime
    submitted: int
    succeeded: int
    failed: int
    results: list[BatchItemResult]


# ---------------------------------------------------------------------------
# History, audit and export
# ---------------------------------------------------------------------------

class ConsultationSummary(BaseModel):
    consultation_id: str
    session_id: str
    created_at: datetime
    patient_reference: str | None = None
    source: str
    batch_id: str | None = None
    original_text: str
    predicted_disease: str
    confidence: float
    engine_key: str
    engine_name: str
    abstained: bool


class ConsultationDetail(ConsultationSummary):
    model_config = _ALLOW_MODEL_PREFIX

    cleaned_text: str
    requested_model_key: str
    policy: str
    policy_reason: str
    engine_metrics: EngineMetrics
    extracted_entities: ExtractedEntities
    engine_opinions: list[EngineOpinion]
    recommended_actions: list[str]
    recommended_medicines: list[MedicationRecommendation]
    disclaimer: str


class ConsultationListResponse(BaseModel):
    total: int
    limit: int
    offset: int
    items: list[ConsultationSummary]


class AuditEvent(BaseModel):
    """An immutable record of what was suggested, when, and by which engine."""

    audit_id: str
    consultation_id: str
    created_at: datetime
    event_type: str
    engine_key: str
    engine_name: str
    engine_accuracy: float | None = None
    engine_metrics_basis: str | None = None
    predicted_disease: str
    confidence: float
    suggested_medications: list[str]
    detail: str


class AuditListResponse(BaseModel):
    consultation_id: str
    events: list[AuditEvent]


class ConsultationReport(BaseModel):
    """A consultation rendered as a shareable text report."""

    consultation_id: str
    generated_at: datetime
    format: str
    content: str


class HistoryStatsResponse(BaseModel):
    total_consultations: int
    total_sessions: int
    abstentions: int
    by_disease: dict[str, int]
    by_engine: dict[str, int]
    first_consultation_at: datetime | None = None
    last_consultation_at: datetime | None = None


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    auth_required: bool
    database_ready: bool
    engines_available: dict[str, bool]
