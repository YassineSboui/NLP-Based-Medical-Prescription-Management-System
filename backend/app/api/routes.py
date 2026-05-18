from fastapi import APIRouter

from app.models.schemas import AnalyzeRequest, AnalyzeResponse, HealthResponse, ModelListResponse
from app.services.nlp_service import NLPService


router = APIRouter()
nlp_service = NLPService()


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    return HealthResponse(status="ok", service="medical-prescription-nlp")


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze_symptoms(request: AnalyzeRequest) -> AnalyzeResponse:
    return nlp_service.analyze(request.text, model_key=request.model_key)


@router.get("/models", response_model=ModelListResponse)
def list_models() -> ModelListResponse:
    return ModelListResponse(default_model="classical", models=nlp_service.predictor.list_models())
