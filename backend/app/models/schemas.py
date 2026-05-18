from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=3, description="Patient-reported symptoms in natural language.")
    model_key: str = Field("classical", description="Prediction model to use: classical or advanced.")


class PredictionModelInfo(BaseModel):
    key: str
    name: str
    family: str
    accuracy: float | None = None
    macro_f1: float | None = None
    description: str
    is_default: bool = False
    is_available: bool = True


class ModelListResponse(BaseModel):
    default_model: str
    models: list[PredictionModelInfo]


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


class AnalyzeResponse(BaseModel):
    original_text: str
    cleaned_text: str
    extracted_entities: ExtractedEntities
    predicted_disease: str
    confidence: float
    model_used: PredictionModelInfo
    recommended_actions: list[str]
    recommended_medicines: list[MedicationRecommendation]
    disclaimer: str


class HealthResponse(BaseModel):
    status: str
    service: str
