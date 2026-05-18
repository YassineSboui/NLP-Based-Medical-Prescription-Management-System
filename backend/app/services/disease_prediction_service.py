from __future__ import annotations

import json
import importlib.util

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from app.core.paths import (
    ADVANCED_LABEL_ENCODER_PATH,
    ADVANCED_METADATA_PATH,
    ADVANCED_MODEL_PATH,
    ADVANCED_TOKENIZER_PATH,
    CLASSICAL_METADATA_PATH,
    CLASSICAL_MODEL_PATH,
    DATASET_PATH,
    LEGACY_MODEL_PATH,
)
from app.models.schemas import PredictionModelInfo
from app.utils.text_cleaning import clean_text


class DiseasePredictionService:
    def __init__(self) -> None:
        self.pipeline = self._load_or_train_pipeline()
        self.advanced_model = None
        self.advanced_tokenizer = None
        self.advanced_label_encoder = None
        self.advanced_metadata = self._load_json(ADVANCED_METADATA_PATH)
        self.classical_metadata = self._load_json(CLASSICAL_METADATA_PATH)

    def predict(self, text: str, model_key: str = "classical") -> tuple[str, float, PredictionModelInfo]:
        cleaned = clean_text(text)
        normalized_model_key = model_key if model_key in {"classical", "advanced"} else "classical"

        if normalized_model_key == "advanced" and self._advanced_available():
            prediction, confidence = self._predict_advanced(cleaned)
            return prediction, confidence, self.get_model_info("advanced")

        probabilities = self.pipeline.predict_proba([cleaned])[0]
        classes = self.pipeline.classes_
        best_index = int(probabilities.argmax())
        ml_prediction = str(classes[best_index])
        ml_confidence = float(probabilities[best_index])

        rule_prediction, rule_confidence = self._predict_from_symptom_profiles(cleaned)
        if rule_prediction and rule_confidence >= ml_confidence:
            return rule_prediction, round(rule_confidence, 3), self.get_model_info("classical")

        return ml_prediction, round(ml_confidence, 3), self.get_model_info("classical")

    def list_models(self) -> list[PredictionModelInfo]:
        return [self.get_model_info("classical"), self.get_model_info("advanced")]

    def get_model_info(self, model_key: str) -> PredictionModelInfo:
        if model_key == "advanced":
            return PredictionModelInfo(
                key="advanced",
                name=str(self.advanced_metadata.get("best_model", "Advanced LSTM")),
                family="LSTM sequence model",
                accuracy=self.advanced_metadata.get("accuracy"),
                macro_f1=self.advanced_metadata.get("macro_f1"),
                description="Advanced neural sequence model trained with Keras. Included for comparison; classical remains the default.",
                is_default=False,
                is_available=self._advanced_available(),
            )

        return PredictionModelInfo(
            key="classical",
            name=str(self.classical_metadata.get("best_model", "TF-IDF classifier")),
            family="TF-IDF + classical ML",
            accuracy=self.classical_metadata.get("accuracy"),
            macro_f1=self.classical_metadata.get("macro_f1"),
            description="Default production model used by the app because it has the best validation score and is explainable.",
            is_default=True,
            is_available=True,
        )

    def _advanced_available(self) -> bool:
        return (
            importlib.util.find_spec("tensorflow") is not None
            and ADVANCED_MODEL_PATH.exists()
            and ADVANCED_TOKENIZER_PATH.exists()
            and ADVANCED_LABEL_ENCODER_PATH.exists()
        )

    def _load_advanced_artifacts(self) -> None:
        from tensorflow.keras.models import load_model
        from tensorflow.keras.preprocessing.sequence import pad_sequences

        if self.advanced_model is None:
            self.advanced_model = load_model(ADVANCED_MODEL_PATH)
        if self.advanced_tokenizer is None:
            self.advanced_tokenizer = joblib.load(ADVANCED_TOKENIZER_PATH)
        if self.advanced_label_encoder is None:
            self.advanced_label_encoder = joblib.load(ADVANCED_LABEL_ENCODER_PATH)

    def _predict_advanced(self, cleaned_text: str) -> tuple[str, float]:
        from tensorflow.keras.preprocessing.sequence import pad_sequences

        self._load_advanced_artifacts()
        max_sequence_length = int(self.advanced_metadata.get("max_sequence_length", 48))
        sequence = self.advanced_tokenizer.texts_to_sequences([cleaned_text])
        padded = pad_sequences(sequence, maxlen=max_sequence_length, padding="post", truncating="post")
        probabilities = self.advanced_model.predict(padded, verbose=0)[0]
        best_index = int(probabilities.argmax())
        prediction = str(self.advanced_label_encoder.inverse_transform([best_index])[0])
        return prediction, round(float(probabilities[best_index]), 3)

    @staticmethod
    def _load_json(path) -> dict:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return {}

    def _load_or_train_pipeline(self) -> Pipeline:
        if CLASSICAL_MODEL_PATH.exists():
            return joblib.load(CLASSICAL_MODEL_PATH)
        if LEGACY_MODEL_PATH.exists():
            return joblib.load(LEGACY_MODEL_PATH)

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
