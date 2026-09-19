"""The three prediction engines, each a first-class citizen.

The previous design had a hidden fourth behaviour. ``_predict_from_symptom_
profiles`` -- 14 hand-written symptom sets, substring-scored, with a bespoke
"+2 if it looks like cholera" bonus and a confidence invented as
``min(0.92, 0.35 + score / profile_size)`` -- silently took over whenever that
invented number came out at least as high as the classifier's probability. The
response still said ``model_used: complement_naive_bayes`` with
``accuracy: 0.906``. The UI showed the ML model's metrics for a prediction the
ML model had not made.

Two things were wrong with that, and both are fixed here:

* **Attribution.** Every engine now announces itself. The arbiter in
  ``disease_prediction_service`` reports which engine decided, and the metrics
  it returns belong to that engine.
* **Comparability.** A rule score and a classifier probability are not on the
  same scale, so comparing them with ``>=`` was meaningless. The rule engine's
  confidence is explicitly documented as a match share rather than a
  probability, and the arbitration policy never compares the two numbers
  directly.

The rule engine is no longer hand-written either. Its profiles are read from
the dataset's ``symptom_terms`` column, so they are backed by the same CDC/WHO
passages as everything else, and the cholera special case is gone -- replaced by
an inverse-frequency weight that gives a rare symptom like "koplik spots" more
say than a symptom half the labels share.
"""

from __future__ import annotations

import importlib.util
import json
import math
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd

from app.core.lexicon import cleaned_forms
from app.core.paths import (
    ADVANCED_LABEL_ENCODER_PATH,
    ADVANCED_METADATA_PATH,
    ADVANCED_MODEL_PATH,
    ADVANCED_TOKENIZER_PATH,
    CLASSICAL_METADATA_PATH,
    CLASSICAL_MODEL_PATH,
    DATASET_PATH,
    ENGINE_METRICS_PATH,
    LEGACY_MODEL_PATH,
)

CLASSICAL_ENGINE = "classical"
SEQUENCE_ENGINE = "advanced"
SYMPTOM_PROFILE_ENGINE = "symptom_profile"

# A rule match needs at least this many distinct symptoms before the engine will
# express an opinion at all. One symptom in common with a profile is noise.
MIN_PROFILE_MATCHES = 2


@dataclass(frozen=True)
class EngineOutcome:
    """What one engine thinks, plus enough context to explain itself."""

    engine_key: str
    disease: str | None
    confidence: float | None
    available: bool
    detail: str
    matched_symptoms: tuple[str, ...] = ()


def _load_json(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _engine_metrics_file() -> dict:
    return _load_json(ENGINE_METRICS_PATH)


class PredictionEngine(ABC):
    """Something that can name a disease and say how sure it is."""

    key: str
    name: str
    family: str
    description: str
    confidence_meaning: str

    @abstractmethod
    def is_available(self) -> bool: ...

    @abstractmethod
    def predict(self, cleaned_text: str) -> EngineOutcome: ...

    def metrics(self) -> dict:
        """Measured performance of *this* engine.

        Comes from models/evaluation/engine_metrics.json, which is produced by
        backend/scripts/evaluate_engines.py running every engine over the same
        held-out test set. An engine that has not been evaluated returns empty
        metrics rather than borrowing another engine's numbers.
        """
        return _engine_metrics_file().get("engines", {}).get(self.key, {})


class ClassicalModelEngine(PredictionEngine):
    key = CLASSICAL_ENGINE
    family = "TF-IDF + linear classifier"
    description = (
        "Default engine. TF-IDF features over cleaned symptom text feeding a linear "
        "classifier selected on a held-out validation set. Explainable and cheap."
    )
    confidence_meaning = "predict_proba of the winning class; a probability over the 14 known labels."

    def __init__(self) -> None:
        self._metadata = _load_json(CLASSICAL_METADATA_PATH)
        self._pipeline = self._load_pipeline()

    @property
    def name(self) -> str:
        return str(self._metadata.get("best_model", "tfidf_classifier"))

    def _load_pipeline(self):
        # joblib.load executes pickled code, so it is only ever pointed at
        # artifacts this repository produced itself via backend/scripts/. These
        # paths are fixed in app.core.paths and are never taken from a request.
        if CLASSICAL_MODEL_PATH.exists():
            return joblib.load(CLASSICAL_MODEL_PATH)
        if LEGACY_MODEL_PATH.exists():
            return joblib.load(LEGACY_MODEL_PATH)
        return self._train_fallback()

    @staticmethod
    def _train_fallback():
        """Last resort so a fresh clone still answers before anyone trains.

        Deliberately not the same configuration the training script selects, and
        it reports no metrics, because an untrained-on-purpose fallback should
        not be able to claim a trained model's numbers.
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline

        from app.utils.text_cleaning import clean_text

        dataset = pd.read_csv(DATASET_PATH)
        pipeline = Pipeline(
            steps=[
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
                ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced")),
            ]
        )
        pipeline.fit(dataset["text"].apply(clean_text), dataset["disease"])
        return pipeline

    def is_available(self) -> bool:
        return self._pipeline is not None

    def predict(self, cleaned_text: str) -> EngineOutcome:
        probabilities = self._pipeline.predict_proba([cleaned_text])[0]
        classes = self._pipeline.classes_
        best = int(probabilities.argmax())
        return EngineOutcome(
            engine_key=self.key,
            disease=str(classes[best]),
            confidence=round(float(probabilities[best]), 3),
            available=True,
            detail=f"Highest class probability among {len(classes)} known labels.",
        )

    def metrics(self) -> dict:
        measured = super().metrics()
        if measured:
            return measured
        # Fall back to what the training run recorded, clearly marked.
        if self._metadata:
            return {
                "accuracy": self._metadata.get("accuracy"),
                "accuracy_ci_95": self._metadata.get("accuracy_ci_95"),
                "macro_f1": self._metadata.get("macro_f1"),
                "basis": self._metadata.get("metric_basis", "training run metadata"),
            }
        return {}


class SequenceModelEngine(PredictionEngine):
    key = SEQUENCE_ENGINE
    family = "Bidirectional recurrent sequence model (Keras)"
    description = (
        "Optional experiment. Requires TensorFlow and the artifacts in models/advanced/. "
        "Kept for comparison; it does not beat the classical engine on this dataset."
    )
    confidence_meaning = "softmax output of the winning class; a probability over the 14 known labels."

    def __init__(self) -> None:
        self._metadata = _load_json(ADVANCED_METADATA_PATH)
        self._model = None
        self._tokenizer = None
        self._encoder = None

    @property
    def name(self) -> str:
        return str(self._metadata.get("best_model", "sequence_model"))

    def is_available(self) -> bool:
        return (
            importlib.util.find_spec("tensorflow") is not None
            and ADVANCED_MODEL_PATH.exists()
            and ADVANCED_TOKENIZER_PATH.exists()
            and ADVANCED_LABEL_ENCODER_PATH.exists()
        )

    def _load(self) -> None:
        from tensorflow.keras.models import load_model

        if self._model is None:
            self._model = load_model(ADVANCED_MODEL_PATH)
        if self._tokenizer is None:
            self._tokenizer = joblib.load(ADVANCED_TOKENIZER_PATH)
        if self._encoder is None:
            self._encoder = joblib.load(ADVANCED_LABEL_ENCODER_PATH)

    def predict(self, cleaned_text: str) -> EngineOutcome:
        if not self.is_available():
            return EngineOutcome(
                engine_key=self.key,
                disease=None,
                confidence=None,
                available=False,
                detail="TensorFlow or the models/advanced/ artifacts are not present.",
            )

        from tensorflow.keras.preprocessing.sequence import pad_sequences

        self._load()
        max_length = int(self._metadata.get("max_sequence_length", 48))
        padded = pad_sequences(
            self._tokenizer.texts_to_sequences([cleaned_text]),
            maxlen=max_length,
            padding="post",
            truncating="post",
        )
        probabilities = self._model.predict(padded, verbose=0)[0]
        best = int(probabilities.argmax())
        return EngineOutcome(
            engine_key=self.key,
            disease=str(self._encoder.inverse_transform([best])[0]),
            confidence=round(float(probabilities[best]), 3),
            available=True,
            detail="Highest softmax probability from the sequence model.",
        )

    def metrics(self) -> dict:
        measured = super().metrics()
        if measured:
            return measured
        if self._metadata:
            return {
                "accuracy": self._metadata.get("accuracy"),
                "macro_f1": self._metadata.get("macro_f1"),
                "basis": self._metadata.get("metric_basis", "training run metadata"),
            }
        return {}


class SymptomProfileEngine(PredictionEngine):
    """Transparent rule engine over source-backed symptom profiles.

    Scoring, in full, because an engine that cannot be explained has no business
    deciding anything:

    1. A profile for each disease is the set of canonical symptoms the dataset
       records for it -- which means the set the CDC/WHO passages attest.
    2. Each symptom carries an inverse-frequency weight, ``log(n_labels /
       n_labels_with_this_symptom)``. "Fever" appears in almost every profile
       and is worth little; "koplik spots" appears in one and is worth a lot.
       This replaces the hand-tuned "+2 for cholera" bonus the old code needed.
    3. A disease scores the weight it matched divided by the weight in its whole
       profile, so a short profile is not penalised for being short.
    4. Confidence is that score as a share of every disease's score. It is a
       *relative* match strength, not a probability, and it is labelled as such
       everywhere it is published.
    """

    key = SYMPTOM_PROFILE_ENGINE
    name = "symptom_profile_rules"
    family = "Weighted symptom-profile matching"
    description = (
        "Rule engine. Matches the symptoms it can find in the text against per-disease "
        "symptom profiles derived from the dataset, weighting rarer symptoms more heavily."
    )
    confidence_meaning = (
        "Share of total profile-match weight, NOT a probability. It cannot be compared "
        "directly with a model probability, and the arbitration policy does not try to."
    )

    def __init__(self, profile_source: pd.DataFrame | None = None) -> None:
        """
        ``profile_source`` exists for evaluation. At runtime the engine builds its
        profiles from the whole dataset, which is correct -- the profiles are its
        knowledge base and there is no test set in production. When *measuring*
        it, though, building profiles from rows that include the test split would
        let it see the answers, so evaluation passes train+validation instead.
        """
        self._profiles, self._weights = self._build_profiles(profile_source)
        self._forms = cleaned_forms()

    @staticmethod
    def _build_profiles(profile_source: pd.DataFrame | None = None) -> tuple[dict[str, set[str]], dict[str, float]]:
        dataset = pd.read_csv(DATASET_PATH) if profile_source is None else profile_source
        profiles: dict[str, set[str]] = defaultdict(set)
        for _, row in dataset.iterrows():
            terms = str(row["symptom_terms"]).split("|")
            profiles[str(row["disease"])].update(term for term in terms if term)

        label_count = len(profiles)
        occurrences: dict[str, int] = defaultdict(int)
        for symptoms in profiles.values():
            for symptom in symptoms:
                occurrences[symptom] += 1

        weights = {
            symptom: math.log(label_count / count) + 1.0
            for symptom, count in occurrences.items()
        }
        return dict(profiles), weights

    def is_available(self) -> bool:
        return bool(self._profiles)

    def _find_symptoms(self, cleaned_text: str) -> set[str]:
        found: set[str] = set()
        for canonical, forms in self._forms.items():
            if any(re.search(rf"\b{re.escape(form)}\b", cleaned_text) for form in forms):
                found.add(canonical)
        return found

    def predict(self, cleaned_text: str) -> EngineOutcome:
        mentioned = self._find_symptoms(cleaned_text)
        if not mentioned:
            return EngineOutcome(
                engine_key=self.key,
                disease=None,
                confidence=None,
                available=True,
                detail="No known symptom term was found in the text.",
            )

        scores: dict[str, float] = {}
        matches: dict[str, set[str]] = {}
        for disease, profile in self._profiles.items():
            matched = mentioned & profile
            matches[disease] = matched
            profile_weight = sum(self._weights[symptom] for symptom in profile)
            matched_weight = sum(self._weights[symptom] for symptom in matched)
            scores[disease] = matched_weight / profile_weight if profile_weight else 0.0

        best_disease = max(scores, key=lambda disease: (scores[disease], len(matches[disease])))
        if len(matches[best_disease]) < MIN_PROFILE_MATCHES:
            return EngineOutcome(
                engine_key=self.key,
                disease=None,
                confidence=None,
                available=True,
                detail=(
                    f"Best profile matched only {len(matches[best_disease])} symptom(s); "
                    f"at least {MIN_PROFILE_MATCHES} are required."
                ),
                matched_symptoms=tuple(sorted(matches[best_disease])),
            )

        total = sum(scores.values())
        confidence = scores[best_disease] / total if total else 0.0
        matched = tuple(sorted(matches[best_disease]))
        return EngineOutcome(
            engine_key=self.key,
            disease=best_disease,
            confidence=round(float(confidence), 3),
            available=True,
            detail=f"Matched {len(matched)} profile symptoms: {', '.join(matched)}.",
            matched_symptoms=matched,
        )
