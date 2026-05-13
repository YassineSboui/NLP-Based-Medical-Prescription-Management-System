from __future__ import annotations

import json
from pathlib import Path

from app.models.schemas import MedicationRecommendation


BACKEND_DIR = Path(__file__).resolve().parents[2]
KB_PATH = BACKEND_DIR / "app" / "data" / "medication_knowledge_base.json"


class RecommendationService:
    def __init__(self) -> None:
        with KB_PATH.open("r", encoding="utf-8") as file:
            self.knowledge_base = json.load(file)

    def get_recommendation(self, disease: str) -> dict[str, list]:
        item = self.knowledge_base.get(disease, self.knowledge_base["unknown"])
        medications = [MedicationRecommendation(**medication) for medication in item["medications"]]
        return {"actions": list(item["actions"]), "medications": medications}
