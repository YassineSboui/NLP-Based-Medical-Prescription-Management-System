from __future__ import annotations

import re
from dataclasses import dataclass

from app.core.lexicon import cleaned_forms
from app.models.schemas import Decision, ExtractedEntities, MedicationRecommendation
from app.services.disease_prediction_service import Decision as EngineDecision
from app.services.disease_prediction_service import DiseasePredictionService
from app.services.recommendation_service import RecommendationService
from app.utils.text_cleaning import clean_text


# The symptom vocabulary is shared with the dataset builder and the rule engine
# via app/data/symptom_lexicon.json. It used to be a second, subtly different
# copy maintained here by hand: this file said "pain behind eyes" where the
# dataset said "pain behind the eyes", so the same symptom had two names
# depending on which half of the system you asked.
SYMPTOM_TERMS: dict[str, tuple[str, ...]] = cleaned_forms()

DISEASE_TERMS = {
    "malaria": ["malaria", "paludisme"],
    "typhoid fever": ["typhoid", "typhoid fever", "fievre typhoide", "fièvre typhoïde"],
    "tuberculosis": ["tuberculosis", "tb", "tuberculose"],
    "hiv": ["hiv", "vih", "aids"],
    "flu": ["flu", "influenza", "grippe"],
    "common cold": ["common cold", "cold", "rhume"],
    "gastroenteritis": ["gastroenteritis", "stomach flu", "food poisoning"],
    "covid-like illness": ["covid", "coronavirus", "covid-19"],
    "dengue": ["dengue", "break bone fever", "break-bone fever"],
    "cholera": ["cholera"],
    "pneumonia": ["pneumonia", "lung infection"],
    "meningitis": ["meningitis"],
    "hepatitis b": ["hepatitis b", "hbv"],
    "measles": ["measles", "rubeola"],
}

MEDICATION_TERMS = {
    "paracetamol": ["paracetamol", "acetaminophen"],
    "oral rehydration salts": ["ors", "oral rehydration salts"],
    "artemether-lumefantrine": ["artemether", "lumefantrine", "coartem"],
    "ceftriaxone": ["ceftriaxone"],
    "azithromycin": ["azithromycin"],
    "rifampicin": ["rifampicin", "rifampin"],
    "isoniazid": ["isoniazid"],
    "antiretroviral therapy": ["art", "antiretroviral", "tenofovir", "lamivudine", "dolutegravir"],
    "ibuprofen": ["ibuprofen"],
    "aspirin": ["aspirin"],
    "entecavir": ["entecavir"],
    "vitamin a": ["vitamin a"],
}

DISCLAIMER = (
    "This application is for academic and educational purposes only. It does not provide a medical diagnosis, "
    "does not replace a licensed healthcare professional, and must not be used for self-medication. Confirm all "
    "symptoms, tests, diagnoses, medicines, and dosages with a qualified clinician."
)


@dataclass(frozen=True)
class Analysis:
    """The result of analysing one note, before it is persisted or serialised.

    Deliberately not an ``AnalyzeResponse``: the response also needs a
    consultation id and a session id, and those belong to the request handler,
    not to the analysis.
    """

    original_text: str
    cleaned_text: str
    entities: ExtractedEntities
    decision: EngineDecision
    recommended_actions: list[str]
    recommended_medicines: list[MedicationRecommendation]
    disclaimer: str = DISCLAIMER

    def to_decision_schema(self) -> Decision:
        return Decision(
            predicted_disease=self.decision.predicted_disease,
            confidence=self.decision.confidence,
            decided_by=self.decision.decided_by,
            policy=self.decision.policy,
            policy_reason=self.decision.policy_reason,
            abstained=self.decision.abstained,
        )


class NLPService:
    def __init__(self) -> None:
        self.predictor = DiseasePredictionService()
        self.recommendations = RecommendationService()

    def analyze(self, text: str, model_key: str = "classical") -> Analysis:
        cleaned_text = clean_text(text)
        entities = self.extract_entities(cleaned_text)
        decision = self.predictor.predict(text, model_key=model_key)
        recommendation = self.recommendations.get_recommendation(decision.predicted_disease)
        actions = list(recommendation["actions"])

        if decision.abstained:
            actions.insert(
                0,
                "No engine was confident enough to suggest a condition, so none is being claimed. "
                "Describe the symptoms to a clinician instead of relying on this result.",
            )

        if entities.diseases and entities.diseases[0] != decision.predicted_disease:
            actions.insert(
                0,
                f"You mentioned {entities.diseases[0]}; compare this with clinical tests because "
                f"symptom-only prediction suggested {decision.predicted_disease}.",
            )

        return Analysis(
            original_text=text,
            cleaned_text=cleaned_text,
            entities=entities,
            decision=decision,
            recommended_actions=actions,
            recommended_medicines=recommendation["medications"],
        )

    def extract_entities(self, cleaned_text: str) -> ExtractedEntities:
        return ExtractedEntities(
            symptoms=self._match_terms(cleaned_text, SYMPTOM_TERMS),
            diseases=self._match_terms(cleaned_text, DISEASE_TERMS),
            medications=self._match_terms(cleaned_text, MEDICATION_TERMS),
            dosage_mentions=self._extract_dosage_mentions(cleaned_text),
        )

    @staticmethod
    def _match_terms(text: str, dictionary: dict[str, list[str]]) -> list[str]:
        matches: list[str] = []
        for canonical, aliases in dictionary.items():
            if any(NLPService._has_positive_match(text, alias) for alias in aliases):
                matches.append(canonical)
        return matches

    @staticmethod
    def _has_positive_match(text: str, alias: str) -> bool:
        match = re.search(rf"\b{re.escape(alias)}\b", text)
        if not match:
            return False

        prefix = text[max(0, match.start() - 18) : match.start()]
        return not re.search(r"\b(no|not|without|denies|deny)\s+$", prefix)

    @staticmethod
    def _extract_dosage_mentions(text: str) -> list[str]:
        patterns = [
            r"\b\d+\s?(?:mg|g|ml|mcg|iu)\b",
            r"\b\d+\s?(?:tablet|tablets|capsule|capsules|pills?)\b",
            r"\b(?:once|twice|three times|four times)\s(?:daily|a day|per day)\b",
            r"\b(?:once|twice|three times|four times)\sday\b",
            r"\bevery\s\d+\s(?:hours|hrs|days)\b",
        ]
        mentions: list[str] = []
        for pattern in patterns:
            mentions.extend(re.findall(pattern, text))
        return sorted(set(mentions))
