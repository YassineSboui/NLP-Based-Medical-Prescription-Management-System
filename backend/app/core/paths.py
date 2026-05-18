from __future__ import annotations

from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[2]
PROJECT_DIR = BACKEND_DIR.parent

DATA_DIR = BACKEND_DIR / "app" / "data"
DATASET_PATH = DATA_DIR / "symptoms_dataset.csv"

MODEL_ROOT_DIR = PROJECT_DIR / "models"
CLASSICAL_MODEL_DIR = MODEL_ROOT_DIR / "classical"
ADVANCED_MODEL_DIR = MODEL_ROOT_DIR / "advanced"

CLASSICAL_MODEL_PATH = CLASSICAL_MODEL_DIR / "trained_model.joblib"
CLASSICAL_VECTORIZER_PATH = CLASSICAL_MODEL_DIR / "vectorizer.joblib"
CLASSICAL_METRICS_PATH = CLASSICAL_MODEL_DIR / "metrics.txt"
CLASSICAL_METADATA_PATH = CLASSICAL_MODEL_DIR / "model_metadata.json"
CONFUSION_MATRIX_PATH = CLASSICAL_MODEL_DIR / "confusion_matrix.png"
NORMALIZED_CONFUSION_MATRIX_PATH = CLASSICAL_MODEL_DIR / "confusion_matrix_normalized.png"
MODEL_COMPARISON_PATH = CLASSICAL_MODEL_DIR / "model_comparison.png"
CLASSIFICATION_REPORT_PATH = CLASSICAL_MODEL_DIR / "classification_report.png"

ADVANCED_MODEL_PATH = ADVANCED_MODEL_DIR / "best_sequence_model.keras"
ADVANCED_TOKENIZER_PATH = ADVANCED_MODEL_DIR / "tokenizer.joblib"
ADVANCED_LABEL_ENCODER_PATH = ADVANCED_MODEL_DIR / "label_encoder.joblib"
ADVANCED_METADATA_PATH = ADVANCED_MODEL_DIR / "deep_learning_metadata.json"

LEGACY_MODEL_PATH = MODEL_ROOT_DIR / "trained_model.joblib"
