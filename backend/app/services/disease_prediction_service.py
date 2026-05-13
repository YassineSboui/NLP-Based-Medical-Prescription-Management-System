from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from app.utils.text_cleaning import clean_text


BACKEND_DIR = Path(__file__).resolve().parents[2]
PROJECT_DIR = BACKEND_DIR.parent
DATASET_PATH = BACKEND_DIR / "app" / "data" / "symptoms_dataset.csv"
MODEL_PATH = PROJECT_DIR / "models" / "trained_model.joblib"


class DiseasePredictionService:
    def __init__(self) -> None:
        self.pipeline = self._load_or_train_pipeline()

    def predict(self, text: str) -> tuple[str, float]:
        cleaned = clean_text(text)
        probabilities = self.pipeline.predict_proba([cleaned])[0]
        classes = self.pipeline.classes_
        best_index = int(probabilities.argmax())
        ml_prediction = str(classes[best_index])
        ml_confidence = float(probabilities[best_index])

        rule_prediction, rule_confidence = self._predict_from_symptom_profiles(cleaned)
        if rule_prediction and rule_confidence >= ml_confidence:
            return rule_prediction, round(rule_confidence, 3)

        return ml_prediction, round(ml_confidence, 3)

    def _load_or_train_pipeline(self) -> Pipeline:
        if MODEL_PATH.exists():
            return joblib.load(MODEL_PATH)

        dataset = pd.read_csv(DATASET_PATH)
        dataset["cleaned_text"] = dataset["text"].apply(clean_text)

        pipeline = Pipeline(
            steps=[
                ("tfidf", TfidfVectorizer(ngram_range=(1, 3), min_df=1, sublinear_tf=True)),
                ("classifier", LogisticRegression(max_iter=1500, class_weight="balanced")),
            ]
        )
        pipeline.fit(dataset["cleaned_text"], dataset["disease"])
        return pipeline

    @staticmethod
    def _predict_from_symptom_profiles(text: str) -> tuple[str | None, float]:
        profiles = {
            "malaria": {"fever", "chills", "sweating", "headache", "body pain", "weakness"},
            "typhoid fever": {"fever", "abdominal pain", "diarrhea", "constipation", "headache", "appetite"},
            "tuberculosis": {"persistent cough", "cough", "night sweats", "weight loss", "chest pain", "blood"},
            "hiv": {"fever", "rash", "mouth ulcers", "oral thrush", "weight loss", "recurrent infections"},
            "flu": {"fever", "sore throat", "body pain", "cough", "chills", "fatigue"},
            "common cold": {"runny nose", "blocked nose", "sneezing", "sore throat", "mild cough"},
            "gastroenteritis": {"diarrhea", "vomiting", "abdominal pain", "nausea", "cramps", "stomach pain"},
            "covid-like illness": {"fever", "dry cough", "shortness breath", "loss taste", "loss smell", "fatigue"},
            "dengue": {"high fever", "severe headache", "pain behind eyes", "joint pain", "muscle pain", "rash", "bleeding"},
            "cholera": {"watery diarrhea", "acute watery diarrhea", "dehydration", "unsafe water", "severe diarrhea", "thirst"},
            "pneumonia": {"chest pain", "cough", "shortness breath", "fever", "chills", "confusion"},
            "meningitis": {"fever", "headache", "stiff neck", "light sensitivity", "confusion", "vomiting"},
            "hepatitis b": {"jaundice", "dark urine", "tired", "nausea", "vomiting", "abdominal pain"},
            "measles": {"high fever", "cough", "runny nose", "red watery eyes", "koplik spots", "rash"},
        }

        scores: dict[str, int] = {}
        for disease, symptoms in profiles.items():
            scores[disease] = sum(1 for symptom in symptoms if symptom in text)

        if "watery diarrhea" in text and any(term in text for term in ["dehydration", "unsafe water", "thirst"]):
            scores["cholera"] += 2

        best_disease = max(scores, key=scores.get)
        best_score = scores[best_disease]
        if best_score < 2:
            return None, 0.0

        profile_size = len(profiles[best_disease])
        confidence = min(0.92, 0.35 + (best_score / profile_size))
        return best_disease, confidence
