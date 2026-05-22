from __future__ import annotations

from pathlib import Path
import json
import sys

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.svm import LinearSVC


BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BACKEND_DIR))

from app.core.paths import (  # noqa: E402
    CLASSICAL_METADATA_PATH,
    CLASSICAL_METRICS_PATH,
    CLASSICAL_MODEL_DIR,
    CLASSICAL_MODEL_PATH,
    CLASSICAL_VECTORIZER_PATH,
    CLASSIFICATION_REPORT_PATH,
    CONFUSION_MATRIX_PATH,
    DATASET_PATH,
    MODEL_COMPARISON_PATH,
    NORMALIZED_CONFUSION_MATRIX_PATH,
)
from app.utils.text_cleaning import clean_text  # noqa: E402


# Plot the confusion matrix so we can see which disease labels are confused.
def save_confusion_matrix_plot(labels: list[str], matrix, title: str, output_path: Path, fmt: str) -> None:
    plt.figure(figsize=(13, 10))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=True,
    )
    plt.title(title)
    plt.xlabel("Predicted disease")
    plt.ylabel("Actual disease")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


# Plot the best optimized version of each model family against the others.
def save_model_comparison_plot(results: list[dict[str, float | str]], output_path: Path) -> None:
    result_frame = pd.DataFrame(results).melt(id_vars="model", value_vars=["accuracy", "macro_f1"], var_name="metric", value_name="score")
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=result_frame, x="model", y="score", hue="metric", palette="Set2")
    ax.set_ylim(0, 1)
    ax.set_title("Optimized Model Comparison")
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


# Convert the classification report into a heatmap for presentation/report screenshots.
def save_classification_report_plot(report: dict, labels: list[str], output_path: Path) -> None:
    rows = []
    for label in labels:
        metrics = report[label]
        rows.append(
            {
                "disease": label,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1-score": metrics["f1-score"],
            }
        )

    report_frame = pd.DataFrame(rows).set_index("disease")
    plt.figure(figsize=(10, 8))
    sns.heatmap(report_frame, annot=True, fmt=".2f", cmap="YlGnBu", vmin=0, vmax=1)
    plt.title("Classification Report by Disease")
    plt.xlabel("Metric")
    plt.ylabel("Disease")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    # 1. Load the curated dataset and clean symptom text before feature extraction.
    dataset = pd.read_csv(DATASET_PATH)
    dataset["cleaned_text"] = dataset["text"].apply(clean_text)

    # 2. Keep 25% of the data for validation. Stratification preserves disease balance.
    train_text, test_text, train_labels, test_labels = train_test_split(
        dataset["cleaned_text"],
        dataset["disease"],
        test_size=0.25,
        random_state=42,
        stratify=dataset["disease"],
    )

    # 3. TF-IDF settings to test internally. The final metadata keeps only the best one.
    vectorizer_configs = [
        {"ngram_range": (1, 2), "min_df": 1, "sublinear_tf": True},
        {"ngram_range": (1, 3), "min_df": 1, "sublinear_tf": True},
        {"ngram_range": (1, 4), "min_df": 1, "sublinear_tf": True},
        {"ngram_range": (1, 3), "min_df": 1, "sublinear_tf": False},
        {"ngram_range": (1, 3), "min_df": 2, "sublinear_tf": True},
        {"ngram_range": (1, 3), "min_df": 1, "sublinear_tf": True, "analyzer": "char_wb", "max_features": 2500},
    ]

    # 4. Classical model families and parameter values to optimize.
    # The final comparison is model-to-model, not parameter-to-parameter.
    candidates = [
        ("logistic_regression", LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5), {"C": 0.5}),
        ("logistic_regression", LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0), {"C": 1.0}),
        ("logistic_regression", LogisticRegression(max_iter=2000, class_weight="balanced", C=2.0), {"C": 2.0}),
        ("linear_svc_calibrated", CalibratedClassifierCV(LinearSVC(class_weight="balanced", dual="auto", C=0.5)), {"C": 0.5}),
        ("linear_svc_calibrated", CalibratedClassifierCV(LinearSVC(class_weight="balanced", dual="auto", C=1.0)), {"C": 1.0}),
        ("linear_svc_calibrated", CalibratedClassifierCV(LinearSVC(class_weight="balanced", dual="auto", C=2.0)), {"C": 2.0}),
        ("complement_naive_bayes", ComplementNB(alpha=0.2), {"alpha": 0.2}),
        ("complement_naive_bayes", ComplementNB(alpha=0.4), {"alpha": 0.4}),
        ("complement_naive_bayes", ComplementNB(alpha=0.7), {"alpha": 0.7}),
        ("complement_naive_bayes", ComplementNB(alpha=1.0), {"alpha": 1.0}),
        ("rbf_svc", SVC(C=2.0, kernel="rbf", gamma="scale", probability=True, class_weight="balanced"), {"C": 2.0}),
        ("rbf_svc", SVC(C=5.0, kernel="rbf", gamma="scale", probability=True, class_weight="balanced"), {"C": 5.0}),
    ]

    # 5. Track every internal run, the best version per model family, and the best overall model.
    tuning_results = []
    best_by_model: dict[str, dict] = {}
    best_name = ""
    best_pipeline: Pipeline | None = None
    best_predictions = None
    best_score = -1.0
    best_accuracy = -1.0

    for vectorizer_config in vectorizer_configs:
        for name, classifier, classifier_params in candidates:
            # 6. A pipeline guarantees the same TF-IDF transformation is used for training and prediction.
            pipeline = Pipeline(
                steps=[
                    ("tfidf", TfidfVectorizer(**vectorizer_config)),
                    ("classifier", classifier),
                ]
            )
            # 7. This is the actual training step for one model/vectorizer configuration.
            pipeline.fit(train_text, train_labels)
            predictions = pipeline.predict(test_text)

            # 8. Evaluate using accuracy and macro F1. Macro F1 treats every disease equally.
            accuracy = accuracy_score(test_labels, predictions)
            macro_f1 = f1_score(test_labels, predictions, average="macro", zero_division=0)
            result = {
                "model": name,
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "classifier_params": classifier_params,
                "vectorizer": vectorizer_config,
            }
            tuning_results.append(result)

            # 9. Keep only the best parameter version for each model family.
            model_best = best_by_model.get(name)
            if model_best is None or (macro_f1, accuracy) > (model_best["macro_f1"], model_best["accuracy"]):
                best_by_model[name] = {
                    **result,
                    "pipeline": pipeline,
                    "predictions": predictions,
                }

            # 10. Select the best overall model by macro F1, then accuracy as tie-breaker.
            ranking_score = (macro_f1, accuracy)
            if ranking_score > (best_score, best_accuracy):
                best_name = name
                best_pipeline = pipeline
                best_predictions = predictions
                best_score = macro_f1
                best_accuracy = accuracy

    if best_pipeline is None or best_predictions is None:
        raise RuntimeError("No model candidate was trained.")

    # 11. Prepare a clean model-level comparison for metrics and plots.
    model_results = [
        {
            "model": name,
            "accuracy": item["accuracy"],
            "macro_f1": item["macro_f1"],
            "best_params": {
                "classifier": item["classifier_params"],
                "vectorizer": item["vectorizer"],
            },
        }
        for name, item in sorted(best_by_model.items(), key=lambda pair: (pair[1]["macro_f1"], pair[1]["accuracy"]), reverse=True)
    ]

    # 12. Generate evaluation objects for the best overall model.
    labels = sorted(dataset["disease"].unique().tolist())
    cm = confusion_matrix(test_labels, best_predictions, labels=labels)
    cm_normalized = confusion_matrix(test_labels, best_predictions, labels=labels, normalize="true")
    report_dict = classification_report(test_labels, best_predictions, labels=labels, zero_division=0, output_dict=True)

    # 13. Write a human-readable metrics report.
    metrics = [
        f"Best model: {best_name}",
        f"Accuracy: {accuracy_score(test_labels, best_predictions):.3f}",
        f"Macro F1: {f1_score(test_labels, best_predictions, average='macro', zero_division=0):.3f}",
        "",
        "Optimized model comparison:",
        *[
            f"- {item['model']}: accuracy={item['accuracy']:.3f}, macro_f1={item['macro_f1']:.3f}, best_params={item['best_params']}"
            for item in model_results
        ],
        "",
        classification_report(test_labels, best_predictions, labels=labels, zero_division=0),
    ]

    # 14. Save visual evaluation artifacts for the report and presentation.
    CLASSICAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    save_confusion_matrix_plot(labels, cm, "Confusion Matrix", CONFUSION_MATRIX_PATH, "d")
    save_confusion_matrix_plot(labels, cm_normalized, "Normalized Confusion Matrix", NORMALIZED_CONFUSION_MATRIX_PATH, ".2f")
    save_model_comparison_plot(model_results, MODEL_COMPARISON_PATH)
    save_classification_report_plot(report_dict, labels, CLASSIFICATION_REPORT_PATH)

    # 15. Retrain the selected best pipeline on the full dataset for final app usage.
    best_pipeline.fit(dataset["cleaned_text"], dataset["disease"])

    # 16. Persist the model, vectorizer, metrics, and clean metadata.
    joblib.dump(best_pipeline, CLASSICAL_MODEL_PATH)
    joblib.dump(best_pipeline.named_steps["tfidf"], CLASSICAL_VECTORIZER_PATH)
    CLASSICAL_METRICS_PATH.write_text("\n".join(metrics), encoding="utf-8")
    CLASSICAL_METADATA_PATH.write_text(
        json.dumps(
            {
                "best_model": best_name,
                "accuracy": accuracy_score(test_labels, best_predictions),
                "macro_f1": f1_score(test_labels, best_predictions, average="macro", zero_division=0),
                "best_params": next(item["best_params"] for item in model_results if item["model"] == best_name),
                "dataset_rows": int(len(dataset)),
                "label_count": int(dataset["disease"].nunique()),
                "labels": sorted(dataset["disease"].unique().tolist()),
                "note": "Internal hyperparameter variants were tested during training; only the final selected model is shown here for a clean project submission.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Saved model to {CLASSICAL_MODEL_PATH}")
    print(f"Saved vectorizer to {CLASSICAL_VECTORIZER_PATH}")
    print(f"Saved metrics to {CLASSICAL_METRICS_PATH}")
    print(f"Saved metadata to {CLASSICAL_METADATA_PATH}")
    print(f"Saved confusion matrix to {CONFUSION_MATRIX_PATH}")
    print(f"Saved normalized confusion matrix to {NORMALIZED_CONFUSION_MATRIX_PATH}")
    print(f"Saved model comparison plot to {MODEL_COMPARISON_PATH}")
    print(f"Saved classification report plot to {CLASSIFICATION_REPORT_PATH}")
    print(metrics[0])
    print(metrics[1])


if __name__ == "__main__":
    main()
