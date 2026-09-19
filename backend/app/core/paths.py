from __future__ import annotations

from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[2]
PROJECT_DIR = BACKEND_DIR.parent

TESTS_DIR = PROJECT_DIR / "tests"
USE_CASES_PATH = TESTS_DIR / "use_case_tests.txt"

DATA_DIR = BACKEND_DIR / "app" / "data"
DATASET_PATH = DATA_DIR / "symptoms_dataset.csv"
DATASET_SOURCES_PATH = DATA_DIR / "dataset_sources.json"
SCRAPED_SOURCES_PATH = DATA_DIR / "scraped_medical_sources.json"
KNOWLEDGE_BASE_PATH = DATA_DIR / "medication_knowledge_base.json"
SYMPTOM_LEXICON_PATH = DATA_DIR / "symptom_lexicon.json"

MODEL_ROOT_DIR = PROJECT_DIR / "models"
CLASSICAL_MODEL_DIR = MODEL_ROOT_DIR / "classical"
ADVANCED_MODEL_DIR = MODEL_ROOT_DIR / "advanced"
EVALUATION_DIR = MODEL_ROOT_DIR / "evaluation"

# Every engine's metrics, measured on one shared held-out split, in one file.
# The API reads an engine's numbers from here so that it can never report one
# engine's score for another engine's answer.
ENGINE_METRICS_PATH = EVALUATION_DIR / "engine_metrics.json"

CLASSICAL_MODEL_PATH = CLASSICAL_MODEL_DIR / "trained_model.joblib"
CLASSICAL_VECTORIZER_PATH = CLASSICAL_MODEL_DIR / "vectorizer.joblib"
CLASSICAL_METRICS_PATH = CLASSICAL_MODEL_DIR / "metrics.txt"
CLASSICAL_METADATA_PATH = CLASSICAL_MODEL_DIR / "model_metadata.json"
MODEL_COMPARISON_PATH = CLASSICAL_MODEL_DIR / "model_comparison.png"
OVERALL_METRICS_PATH = CLASSICAL_MODEL_DIR / "overall_metrics.png"
PREDICTION_OUTCOMES_PATH = CLASSICAL_MODEL_DIR / "prediction_outcomes.png"

# One artifact, not three. `best_sequence_model.keras` used to be a byte-for-byte
# copy of `lstm_model.keras`, and `gru_model.keras` was a losing candidate that
# nothing loaded -- 13 MB of binaries in git for one model that gets used.
ADVANCED_MODEL_PATH = ADVANCED_MODEL_DIR / "sequence_model.keras"
ADVANCED_TOKENIZER_PATH = ADVANCED_MODEL_DIR / "tokenizer.joblib"
ADVANCED_LABEL_ENCODER_PATH = ADVANCED_MODEL_DIR / "label_encoder.joblib"
ADVANCED_METADATA_PATH = ADVANCED_MODEL_DIR / "deep_learning_metadata.json"
ADVANCED_METRICS_PATH = ADVANCED_MODEL_DIR / "deep_learning_metrics.txt"
ADVANCED_OVERALL_METRICS_PATH = ADVANCED_MODEL_DIR / "advanced_overall_metrics.png"
ADVANCED_TRAINING_HISTORY_PATH = ADVANCED_MODEL_DIR / "training_history.png"

LEGACY_MODEL_PATH = MODEL_ROOT_DIR / "trained_model.joblib"
