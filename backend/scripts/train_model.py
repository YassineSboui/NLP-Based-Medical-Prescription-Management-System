from __future__ import annotations

from pathlib import Path
import json
import sys

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV


BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = BACKEND_DIR.parent
sys.path.append(str(BACKEND_DIR))

from app.utils.text_cleaning import clean_text  # noqa: E402


DATASET_PATH = BACKEND_DIR / "app" / "data" / "symptoms_dataset.csv"
MODEL_DIR = PROJECT_DIR / "models"
MODEL_PATH = MODEL_DIR / "trained_model.joblib"
VECTORIZER_PATH = MODEL_DIR / "vectorizer.joblib"
REPORT_PATH = MODEL_DIR / "metrics.txt"
METADATA_PATH = MODEL_DIR / "model_metadata.json"


def main() -> None:
    dataset = pd.read_csv(DATASET_PATH)
    dataset["cleaned_text"] = dataset["text"].apply(clean_text)

    train_text, test_text, train_labels, test_labels = train_test_split(
        dataset["cleaned_text"],
        dataset["disease"],
        test_size=0.25,
        random_state=42,
        stratify=dataset["disease"],
    )

    candidates = {
        "logistic_regression": LogisticRegression(max_iter=1500, class_weight="balanced"),
        "linear_svc_calibrated": CalibratedClassifierCV(LinearSVC(class_weight="balanced", dual="auto")),
        "complement_naive_bayes": ComplementNB(alpha=0.4),
    }

    results = []
    best_name = ""
    best_pipeline: Pipeline | None = None
    best_predictions = None
    best_score = -1.0
    best_accuracy = -1.0

    for name, classifier in candidates.items():
        pipeline = Pipeline(
            steps=[
                ("tfidf", TfidfVectorizer(ngram_range=(1, 3), min_df=1, sublinear_tf=True)),
                ("classifier", classifier),
            ]
        )
        pipeline.fit(train_text, train_labels)
        predictions = pipeline.predict(test_text)
        accuracy = accuracy_score(test_labels, predictions)
        macro_f1 = f1_score(test_labels, predictions, average="macro", zero_division=0)
        results.append({"model": name, "accuracy": accuracy, "macro_f1": macro_f1})

        ranking_score = (macro_f1, accuracy)
        if ranking_score > (best_score, best_accuracy):
            best_name = name
            best_pipeline = pipeline
            best_predictions = predictions
            best_score = macro_f1
            best_accuracy = accuracy

    if best_pipeline is None or best_predictions is None:
        raise RuntimeError("No model candidate was trained.")

    metrics = [
        f"Best model: {best_name}",
        f"Accuracy: {accuracy_score(test_labels, best_predictions):.3f}",
        f"Macro F1: {f1_score(test_labels, best_predictions, average='macro', zero_division=0):.3f}",
        "",
        "Candidate comparison:",
        *[f"- {item['model']}: accuracy={item['accuracy']:.3f}, macro_f1={item['macro_f1']:.3f}" for item in results],
        "",
        classification_report(test_labels, best_predictions, zero_division=0),
    ]

    best_pipeline.fit(dataset["cleaned_text"], dataset["disease"])
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(best_pipeline, MODEL_PATH)
    joblib.dump(best_pipeline.named_steps["tfidf"], VECTORIZER_PATH)
    REPORT_PATH.write_text("\n".join(metrics), encoding="utf-8")
    METADATA_PATH.write_text(
        json.dumps(
            {
                "best_model": best_name,
                "dataset_rows": int(len(dataset)),
                "label_count": int(dataset["disease"].nunique()),
                "labels": sorted(dataset["disease"].unique().tolist()),
                "candidate_results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved vectorizer to {VECTORIZER_PATH}")
    print(f"Saved metrics to {REPORT_PATH}")
    print(f"Saved metadata to {METADATA_PATH}")
    print(metrics[0])
    print(metrics[1])


if __name__ == "__main__":
    main()
