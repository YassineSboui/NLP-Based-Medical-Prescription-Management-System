from __future__ import annotations

import re

from app.models.schemas import AnalyzeResponse, ExtractedEntities
from app.services.disease_prediction_service import DiseasePredictionService
from app.services.recommendation_service import RecommendationService
from app.utils.text_cleaning import clean_text


SYMPTOM_TERMS = {
    "fever": ["fever", "high fever", "low grade fever", "temperature"],
    "headache": ["headache", "head pain"],
    "chills": ["chills", "shivering"],
    "night sweats": ["night sweats"],
    "sweating": ["sweating", "sweat"],
    "body pain": ["body pain", "body ache", "aches"],
    "muscle pain": ["muscle pain", "joint pain", "joint pains"],
    "fatigue": ["fatigue", "tired", "weakness", "weak"],
    "abdominal pain": ["abdominal pain", "stomach pain", "belly pain"],
    "diarrhea": ["diarrhea", "diarrhoea", "loose stool", "loose stools", "liquid stools"],
    "watery diarrhea": ["watery diarrhea", "watery stool", "acute watery diarrhea"],
    "dehydration": ["dehydration", "dehydrated", "thirst"],
    "nausea": ["nausea", "nauseous"],
    "vomiting": ["vomiting", "vomit", "persistent vomiting"],
    "cough": ["cough", "coughing"],
    "persistent cough": ["persistent cough", "cough for weeks", "cough weeks", "long cough"],
    "chest pain": ["chest pain", "chest discomfort", "breathing pain"],
    "blood in sputum": ["coughing blood", "cough blood", "blood sputum", "blood phlegm"],
    "weight loss": ["weight loss", "losing weight"],
    "sore throat": ["sore throat", "throat pain"],
    "runny nose": ["runny nose", "blocked nose", "nasal congestion", "sneezing"],
    "shortness of breath": ["shortness of breath", "shortness breath", "difficulty breathing", "breathless"],
    "loss of taste": ["loss of taste", "loss taste", "lost taste"],
    "loss of smell": ["loss smell", "loss of smell", "lost smell"],
    "rash": ["rash", "skin rash"],
    "red watery eyes": ["red watery eyes", "watery eyes", "red eyes"],
    "koplik spots": ["koplik spots", "white spots mouth"],
    "stiff neck": ["stiff neck"],
    "light sensitivity": ["light sensitivity", "sensitive light"],
    "confusion": ["confusion", "altered mental status", "mental confusion"],
    "jaundice": ["jaundice", "yellowing skin", "yellow eyes"],
    "dark urine": ["dark urine", "bloody urine"],
    "bleeding": ["bleeding", "bleeding gums", "nose bleeding", "blood stool", "blood vomit"],
    "mouth ulcers": ["mouth ulcers", "oral thrush", "white patches in mouth"],
    "loss of appetite": ["loss appetite", "loss of appetite", "no appetite", "poor appetite"],
    "swollen glands": ["swollen glands", "swollen lymph nodes"],
    "pain behind eyes": ["pain behind eyes", "pain behind the eyes"],
}

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


class NLPService:
    def __init__(self) -> None:
        self.predictor = DiseasePredictionService()
        self.recommendations = RecommendationService()

    def analyze(self, text: str) -> AnalyzeResponse:
        cleaned_text = clean_text(text)
        entities = self.extract_entities(cleaned_text)
        predicted_disease, confidence = self.predictor.predict(cleaned_text)
        recommendation = self.recommendations.get_recommendation(predicted_disease)

        if entities.diseases and entities.diseases[0] != predicted_disease:
            recommendation["actions"].insert(
                0,
                f"You mentioned {entities.diseases[0]}; compare this with clinical tests because symptom-only prediction suggested {predicted_disease}.",
            )

        return AnalyzeResponse(
            original_text=text,
            cleaned_text=cleaned_text,
            extracted_entities=entities,
            predicted_disease=predicted_disease,
            confidence=confidence,
            recommended_actions=recommendation["actions"],
            recommended_medicines=recommendation["medications"],
            disclaimer=DISCLAIMER,
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
