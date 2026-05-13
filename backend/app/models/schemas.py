from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=3, description="Patient-reported symptoms in natural language.")


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
    recommended_actions: list[str]
    recommended_medicines: list[MedicationRecommendation]
    disclaimer: str


class HealthResponse(BaseModel):
    status: str
    service: str
