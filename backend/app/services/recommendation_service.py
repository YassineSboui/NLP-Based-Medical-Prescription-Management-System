from __future__ import annotations

import json

from app.core.paths import KNOWLEDGE_BASE_PATH
from app.models.schemas import MedicationRecommendation


class RecommendationService:
    """Educational medication information, keyed by predicted condition.

    The ``unknown`` entry is not a fallback of last resort but a real answer:
    when the ensemble abstains, this is what a caller gets, and it says to see a
    clinician rather than naming a drug.
    """

    def __init__(self) -> None:
        with KNOWLEDGE_BASE_PATH.open("r", encoding="utf-8") as file:
            self.knowledge_base = json.load(file)

    def get_recommendation(self, disease: str) -> dict[str, list]:
        item = self.knowledge_base.get(disease, self.knowledge_base["unknown"])
        medications = [MedicationRecommendation(**medication) for medication in item["medications"]]
        return {"actions": list(item["actions"]), "medications": medications}
