"""Train and honestly evaluate the classical symptom classifier.

What was wrong with the previous version of this script
-------------------------------------------------------
It swept 6 vectorizer configs x 12 classifiers = 72 pipelines, scored all 72 on
the *same 32-row split*, published the winner's score on that split as the
headline number, and then refit the winner on all 128 rows including those 32.
Three separate problems, all pushing the same way:

1. Selecting the best of 72 on a 32-row set and then reporting that set's score
   is reporting the maximum of 72 noisy estimates. It is biased upward by
   construction, and 0.906 on 32 rows is +-0.10 before you even start.
2. The shipped artifact had seen every row it was scored on, so the published
   number did not describe the shipped model at all.
3. Near-duplicate rows could sit on both sides of the split.

What this version does instead
------------------------------
* Splits by ``group_id``, not by row. A paraphrase and its parent are
  near-duplicates; with a row-wise split one lands in train and the other in
  test and the score is meaningless. Groups keep them together.
* Three-way split: train / validation / test. Model selection reads the
  validation set only. The test set is opened once, at the end, by one model.
* Stratified group k-fold CV over train+validation for the selected config,
  because a single split of a dataset this size is too noisy to quote alone.
* Publishes intervals, not just point estimates: a t-interval across CV folds,
  and Wilson intervals on per-class recall.
* Ships the model fit on train+validation. It never sees the test set, so the
  published test number describes exactly the artifact in ``models/classical/``
  rather than a differently-fit sibling of it. That costs ~20% of the data, and
  is worth it to make the headline number literally true.
* Adds two diagnostics that matter more than the headline for this dataset:
  an external check on hand-written scenarios, and a cross-source transfer test.

Read ``metrics.txt`` or the ``evaluation`` block of ``model_metadata.json`` for
the methodology alongside the numbers.
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.calibration import CalibratedClassifierCV  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402
from sklearn.naive_bayes import ComplementNB  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.svm import SVC, LinearSVC  # noqa: E402

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BACKEND_DIR))

from app.core.paths import (  # noqa: E402
    CLASSICAL_METADATA_PATH,
    CLASSICAL_METRICS_PATH,
    CLASSICAL_MODEL_DIR,
    CLASSICAL_MODEL_PATH,
    CLASSICAL_VECTORIZER_PATH,
    DATASET_PATH,
    MODEL_COMPARISON_PATH,
    OVERALL_METRICS_PATH,
    PREDICTION_OUTCOMES_PATH,
)
from app.core.use_cases import load_use_cases  # noqa: E402
from app.utils.text_cleaning import clean_text  # noqa: E402

RANDOM_STATE = 42
CV_FOLDS = 5
TEST_FRACTION_FOLDS = 5  # 1/5 of the groups become the test set
VALIDATION_FRACTION_FOLDS = 4  # then 1/4 of what is left becomes validation


# --------------------------------------------------------------------------
# Interval estimation
# --------------------------------------------------------------------------

def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion.

    Wilson rather than the normal approximation because this dataset produces
    per-class supports in the teens, where the normal interval misbehaves and
    can even run outside [0, 1] at proportions near 0 or 1.
    """
    if total == 0:
        return (0.0, 0.0)
    proportion = successes / total
    denominator = 1 + z**2 / total
    centre = proportion + z**2 / (2 * total)
    spread = z * math.sqrt(proportion * (1 - proportion) / total + z**2 / (4 * total**2))
    return (max(0.0, (centre - spread) / denominator), min(1.0, (centre + spread) / denominator))


def t_interval(values: list[float], z: float = 2.776) -> tuple[float, float]:
    """95% interval across CV folds. Default z is t(0.975, df=4) for 5 folds."""
    if len(values) < 2:
        return (float(values[0]) if values else 0.0, float(values[0]) if values else 0.0)
    mean = float(np.mean(values))
    standard_error = float(np.std(values, ddof=1) / math.sqrt(len(values)))
    return (max(0.0, mean - z * standard_error), min(1.0, mean + z * standard_error))


# --------------------------------------------------------------------------
# Splitting
# --------------------------------------------------------------------------

def group_holdout(
    frame: pd.DataFrame,
    n_folds: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carve 1/n_folds of the *groups* out of ``frame``, stratified by label."""
    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    remainder_index, holdout_index = next(
        splitter.split(frame["cleaned_text"], frame["disease"], groups=frame["group_id"])
    )
    return frame.iloc[remainder_index].copy(), frame.iloc[holdout_index].copy()


def assert_no_group_overlap(*frames: pd.DataFrame) -> None:
    """A split that leaks a group is worse than no split; fail loudly."""
    seen: set[str] = set()
    for frame in frames:
        groups = set(frame["group_id"])
        overlap = seen & groups
        if overlap:
            raise RuntimeError(f"Group leak across splits: {sorted(overlap)[:5]}")
        seen |= groups


# --------------------------------------------------------------------------
# Candidate space
# --------------------------------------------------------------------------

def vectorizer_configs() -> list[dict]:
    return [
        {"ngram_range": (1, 2), "min_df": 1, "sublinear_tf": True},
        {"ngram_range": (1, 3), "min_df": 1, "sublinear_tf": True},
        {"ngram_range": (1, 3), "min_df": 2, "sublinear_tf": True},
        {"ngram_range": (1, 2), "min_df": 2, "sublinear_tf": True},
        {"ngram_range": (1, 1), "min_df": 1, "sublinear_tf": True},
        {"ngram_range": (3, 5), "min_df": 2, "sublinear_tf": True, "analyzer": "char_wb", "max_features": 4000},
    ]


def classifier_candidates() -> list[tuple[str, object, dict]]:
    return [
        ("logistic_regression", LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5), {"C": 0.5}),
        ("logistic_regression", LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0), {"C": 1.0}),
        ("logistic_regression", LogisticRegression(max_iter=2000, class_weight="balanced", C=4.0), {"C": 4.0}),
        ("linear_svc_calibrated", CalibratedClassifierCV(LinearSVC(class_weight="balanced", dual="auto", C=0.5)), {"C": 0.5}),
        ("linear_svc_calibrated", CalibratedClassifierCV(LinearSVC(class_weight="balanced", dual="auto", C=1.0)), {"C": 1.0}),
        ("complement_naive_bayes", ComplementNB(alpha=0.2), {"alpha": 0.2}),
        ("complement_naive_bayes", ComplementNB(alpha=0.5), {"alpha": 0.5}),
        ("complement_naive_bayes", ComplementNB(alpha=1.0), {"alpha": 1.0}),
        ("rbf_svc", SVC(C=2.0, kernel="rbf", gamma="scale", probability=True, class_weight="balanced", random_state=RANDOM_STATE), {"C": 2.0}),
        ("rbf_svc", SVC(C=8.0, kernel="rbf", gamma="scale", probability=True, class_weight="balanced", random_state=RANDOM_STATE), {"C": 8.0}),
    ]


def build_pipeline(vectorizer_config: dict, classifier) -> Pipeline:
    from sklearn.base import clone

    return Pipeline(
        steps=[("tfidf", TfidfVectorizer(**vectorizer_config)), ("classifier", clone(classifier))]
    )


# --------------------------------------------------------------------------
# Plots (kept from the previous pipeline; now fed honest numbers)
# --------------------------------------------------------------------------

def save_model_comparison_plot(results: list[dict], output_path: Path) -> None:
    frame = pd.DataFrame(results).melt(
        id_vars="model", value_vars=["accuracy", "macro_f1"], var_name="metric", value_name="score"
    )
    plt.figure(figsize=(10, 6))
    axis = sns.barplot(data=frame, x="model", y="score", hue="metric", palette="Set2")
    axis.set_ylim(0, 1.05)
    axis.set_title("Best configuration per model family (validation set)")
    axis.set_xlabel("Model family")
    axis.set_ylabel("Score")
    for container in axis.containers:
        axis.bar_label(container, fmt="%.3f", padding=3, fontsize=8, fontweight="bold")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def calculate_multiclass_outcomes(actual, predicted, labels: list[str]) -> dict[str, int]:
    matrix = confusion_matrix(actual, predicted, labels=labels)
    total = int(matrix.sum())
    outcomes = {"true_positive": 0, "false_positive": 0, "true_negative": 0, "false_negative": 0}

    for index in range(len(labels)):
        true_positive = int(matrix[index, index])
        false_positive = int(matrix[:, index].sum() - true_positive)
        false_negative = int(matrix[index, :].sum() - true_positive)
        outcomes["true_positive"] += true_positive
        outcomes["false_positive"] += false_positive
        outcomes["false_negative"] += false_negative
        outcomes["true_negative"] += total - true_positive - false_positive - false_negative

    return outcomes


def save_prediction_outcomes_plot(results: list[dict], output_path: Path) -> None:
    frame = pd.DataFrame(results).sort_values("macro_f1", ascending=False).head(4)
    figure, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for axis, row in zip(axes, frame.itertuples()):
        matrix = np.array(
            [[row.true_positive, row.false_positive], [row.false_negative, row.true_negative]]
        )
        annotations = np.array(
            [
                [f"TP\n{row.true_positive}", f"FP\n{row.false_positive}"],
                [f"FN\n{row.false_negative}", f"TN\n{row.true_negative}"],
            ]
        )
        sns.heatmap(
            matrix, annot=annotations, fmt="", cmap="YlGnBu", cbar=False, linewidths=2,
            linecolor="white", square=True, ax=axis, annot_kws={"fontsize": 14, "fontweight": "bold"},
        )
        axis.set_title(f"{row.model}\nAcc {row.accuracy:.3f} | Macro F1 {row.macro_f1:.3f}", fontweight="bold")
        axis.set_xlabel("Predicted")
        axis.set_ylabel("Actual")
        axis.set_xticklabels(["Positive", "Negative"])
        axis.set_yticklabels(["Positive", "Negative"], rotation=0)

    for axis in axes[len(frame):]:
        axis.axis("off")

    figure.suptitle("Validation outcomes by model family (one-vs-rest aggregated)", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_overall_metrics_plot(summary: dict, output_path: Path) -> None:
    """Point estimates with their intervals drawn on, so the plot cannot mislead."""
    names = ["Test accuracy", "Test macro F1", "CV accuracy", "External scenarios"]
    values = [
        summary["test"]["accuracy"],
        summary["test"]["macro_f1"],
        summary["cross_validation"]["accuracy_mean"],
        summary["external_use_cases"]["accuracy"],
    ]
    lows = [
        summary["test"]["accuracy_ci"][0],
        summary["test"]["macro_f1"],
        summary["cross_validation"]["accuracy_ci"][0],
        summary["external_use_cases"]["accuracy_ci"][0],
    ]
    highs = [
        summary["test"]["accuracy_ci"][1],
        summary["test"]["macro_f1"],
        summary["cross_validation"]["accuracy_ci"][1],
        summary["external_use_cases"]["accuracy_ci"][1],
    ]

    plt.figure(figsize=(9, 5.5))
    axis = sns.barplot(x=names, y=values, hue=names, palette="Set2", legend=False)
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Score")
    axis.set_title("Headline metrics with 95% intervals")
    errors = [
        [value - low for value, low in zip(values, lows)],
        [high - value for value, high in zip(values, highs)],
    ]
    axis.errorbar(x=range(len(values)), y=values, yerr=errors, fmt="none", ecolor="black", capsize=6, linewidth=1.4)
    for index, value in enumerate(values):
        axis.text(index, min(1.02, highs[index] + 0.02), f"{value:.3f}", ha="center", fontweight="bold")
    plt.xticks(rotation=12, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


# --------------------------------------------------------------------------
# Diagnostics
# --------------------------------------------------------------------------

def cross_source_transfer(dataset: pd.DataFrame, vectorizer_config: dict, classifier) -> dict:
    """Hold out one entire source passage per label and see what survives.

    This is the number that actually answers "has it learned the disease, or the
    wording of one CDC page?". Labels backed by a single passage cannot be
    tested this way and are reported as not evaluable rather than being quietly
    folded into an average.
    """
    results: dict[str, object] = {}
    per_label: dict[str, dict] = {}
    passages_per_label = dataset.groupby("disease")["source_section_key"].nunique()

    correct_total = 0
    count_total = 0

    for label, passage_count in passages_per_label.items():
        if passage_count < 2:
            per_label[label] = {"evaluable": False, "reason": "only one source passage backs this label"}
            continue

        label_scores: list[float] = []
        for passage in sorted(dataset.loc[dataset["disease"] == label, "source_section_key"].unique()):
            held_out = dataset[(dataset["disease"] == label) & (dataset["source_section_key"] == passage)]
            training = dataset.drop(held_out.index)
            if training["disease"].nunique() < dataset["disease"].nunique():
                continue

            pipeline = build_pipeline(vectorizer_config, classifier)
            pipeline.fit(training["cleaned_text"], training["disease"])
            predictions = pipeline.predict(held_out["cleaned_text"])
            correct = int((predictions == label).sum())

            label_scores.append(correct / len(held_out))
            correct_total += correct
            count_total += len(held_out)

        per_label[label] = {
            "evaluable": True,
            "passages": int(passage_count),
            "mean_recall_on_held_out_passage": round(float(np.mean(label_scores)), 3) if label_scores else None,
        }

    results["per_label"] = per_label
    results["pooled_recall"] = round(correct_total / count_total, 3) if count_total else None
    results["rows_evaluated"] = count_total
    return results


def confidence_floor_sweep(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    vectorizer_config: dict,
    classifier,
) -> list[dict]:
    """Choose the ensemble's confidence floor on validation, honestly.

    This has to happen here rather than in evaluate_engines.py: the shipped
    artifact is fit on train+validation, so asking it about validation rows
    returns its own training data and every floor scores 1.000. The sweep needs
    a model that has only seen ``train``.

    Simulates the arbitration policy rather than calling it, so the deciding
    engine at each floor is visible.
    """
    from app.services.prediction_engines import SymptomProfileEngine

    pipeline = build_pipeline(vectorizer_config, classifier)
    pipeline.fit(train["cleaned_text"], train["disease"])
    probabilities = pipeline.predict_proba(validation["cleaned_text"])
    classes = pipeline.classes_

    # Profiles from training rows only, for the same reason.
    rule_engine = SymptomProfileEngine(profile_source=train)
    rule_outcomes = [rule_engine.predict(text) for text in validation["cleaned_text"]]
    actual = list(validation["disease"])

    rows: list[dict] = []
    for floor in (0.0, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70):
        predictions: list[str] = []
        decided_by_rule = 0
        abstentions = 0

        for index, outcome in enumerate(rule_outcomes):
            best = int(probabilities[index].argmax())
            model_confidence = float(probabilities[index][best])
            if model_confidence >= floor:
                predictions.append(str(classes[best]))
            elif outcome.disease is not None:
                predictions.append(outcome.disease)
                decided_by_rule += 1
            else:
                predictions.append("unknown")
                abstentions += 1

        rows.append(
            {
                "floor": floor,
                "accuracy": round(float(accuracy_score(actual, predictions)), 3),
                "macro_f1": round(float(f1_score(actual, predictions, average="macro", zero_division=0)), 3),
                "decided_by_rule_engine": decided_by_rule,
                "abstentions": abstentions,
            }
        )

    return rows


def external_use_case_check(pipeline: Pipeline, labels: set[str]) -> dict:
    cases = load_use_cases(labels=labels)
    if not cases:
        return {"count": 0, "accuracy": 0.0, "accuracy_ci": [0.0, 0.0], "failures": []}

    predictions = pipeline.predict([clean_text(case.text) for case in cases])
    correct = sum(1 for case, prediction in zip(cases, predictions) if prediction == case.expected)
    low, high = wilson_interval(correct, len(cases))

    return {
        "count": len(cases),
        "correct": correct,
        "accuracy": round(correct / len(cases), 3),
        "accuracy_ci": [round(low, 3), round(high, 3)],
        "failures": [
            {"text": case.text, "expected": case.expected, "predicted": str(prediction)}
            for case, prediction in zip(cases, predictions)
            if prediction != case.expected
        ],
    }


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> None:
    dataset = pd.read_csv(DATASET_PATH)
    dataset["cleaned_text"] = dataset["text"].apply(clean_text)
    dataset["source_section_key"] = dataset["source_name"] + " :: " + dataset["source_section"]
    labels = sorted(dataset["disease"].unique().tolist())

    # 1. Group-aware three-way split. Test is carved first and set aside.
    development, test = group_holdout(dataset, TEST_FRACTION_FOLDS, RANDOM_STATE)
    train, validation = group_holdout(development, VALIDATION_FRACTION_FOLDS, RANDOM_STATE)
    assert_no_group_overlap(train, validation, test)

    print(f"rows: train={len(train)} validation={len(validation)} test={len(test)}")
    print(
        f"groups: train={train['group_id'].nunique()} "
        f"validation={validation['group_id'].nunique()} test={test['group_id'].nunique()}"
    )

    # 2. Model selection. Fit on train, score on validation. The test frame is
    #    not touched anywhere in this loop.
    selection_results: list[dict] = []
    best_by_family: dict[str, dict] = {}
    best: dict | None = None

    for vectorizer_config in vectorizer_configs():
        for name, classifier, classifier_params in classifier_candidates():
            pipeline = build_pipeline(vectorizer_config, classifier)
            pipeline.fit(train["cleaned_text"], train["disease"])
            predictions = pipeline.predict(validation["cleaned_text"])

            record = {
                "model": name,
                "accuracy": float(accuracy_score(validation["disease"], predictions)),
                "macro_f1": float(f1_score(validation["disease"], predictions, average="macro", zero_division=0)),
                "classifier_params": classifier_params,
                "vectorizer": vectorizer_config,
            }
            selection_results.append(record)

            ranking = (record["macro_f1"], record["accuracy"])
            family_best = best_by_family.get(name)
            if family_best is None or ranking > (family_best["macro_f1"], family_best["accuracy"]):
                best_by_family[name] = {**record, "predictions": predictions}
            if best is None or ranking > (best["macro_f1"], best["accuracy"]):
                best = {**record, "classifier": classifier}

    assert best is not None
    print(f"selected on validation: {best['model']} macro_f1={best['macro_f1']:.3f}")

    # 3. Cross-validation of the selected configuration over train+validation.
    #    A single split of ~120 rows is too noisy to quote by itself.
    cv_accuracies: list[float] = []
    cv_macro_f1s: list[float] = []
    splitter = StratifiedGroupKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    for fold_train_index, fold_test_index in splitter.split(
        development["cleaned_text"], development["disease"], groups=development["group_id"]
    ):
        fold_train = development.iloc[fold_train_index]
        fold_test = development.iloc[fold_test_index]
        pipeline = build_pipeline(best["vectorizer"], best["classifier"])
        pipeline.fit(fold_train["cleaned_text"], fold_train["disease"])
        fold_predictions = pipeline.predict(fold_test["cleaned_text"])
        cv_accuracies.append(float(accuracy_score(fold_test["disease"], fold_predictions)))
        cv_macro_f1s.append(float(f1_score(fold_test["disease"], fold_predictions, average="macro", zero_division=0)))

    # 4. Fit the shipped model on train+validation, then open the test set once.
    final_pipeline = build_pipeline(best["vectorizer"], best["classifier"])
    final_pipeline.fit(development["cleaned_text"], development["disease"])
    test_predictions = final_pipeline.predict(test["cleaned_text"])

    test_accuracy = float(accuracy_score(test["disease"], test_predictions))
    test_macro_f1 = float(f1_score(test["disease"], test_predictions, average="macro", zero_division=0))
    accuracy_low, accuracy_high = wilson_interval(int((test_predictions == test["disease"]).sum()), len(test))

    precision, recall, per_class_f1, support = precision_recall_fscore_support(
        test["disease"], test_predictions, labels=labels, zero_division=0
    )
    per_class = []
    for index, label in enumerate(labels):
        hits = int(round(recall[index] * support[index]))
        low, high = wilson_interval(hits, int(support[index]))
        per_class.append(
            {
                "label": label,
                "support": int(support[index]),
                "precision": round(float(precision[index]), 3),
                "recall": round(float(recall[index]), 3),
                "recall_ci_95": [round(low, 3), round(high, 3)],
                "f1": round(float(per_class_f1[index]), 3),
            }
        )

    # 5. Diagnostics.
    external = external_use_case_check(final_pipeline, set(labels))
    transfer = cross_source_transfer(dataset, best["vectorizer"], best["classifier"])
    floor_sweep = confidence_floor_sweep(train, validation, best["vectorizer"], best["classifier"])

    cv_accuracy_ci = t_interval(cv_accuracies)
    cv_macro_f1_ci = t_interval(cv_macro_f1s)

    evaluation = {
        "methodology": {
            "split": "group-aware train/validation/test, groups = dataset group_id",
            "split_sizes": {"train": len(train), "validation": len(validation), "test": len(test)},
            "split_groups": {
                "train": int(train["group_id"].nunique()),
                "validation": int(validation["group_id"].nunique()),
                "test": int(test["group_id"].nunique()),
            },
            "model_selection": (
                f"{len(selection_results)} pipelines fit on train and ranked on validation macro F1. "
                "The test set is not read during selection."
            ),
            "shipped_model_fit_on": (
                "train + validation only. The test set was never seen by the artifact in "
                "models/classical/, so the test numbers below describe that exact artifact."
            ),
            "cross_validation": f"StratifiedGroupKFold(n_splits={CV_FOLDS}) over train+validation",
            "intervals": (
                "Test accuracy and per-class recall use 95% Wilson score intervals. "
                "Cross-validation uses a 95% t-interval across folds."
            ),
            "known_limitations": [
                "Every row for a label descends from one or two CDC/WHO passages, so folds are "
                "not clinically independent. The cross-source transfer diagnostic is the honest "
                "measure of generalisation; the test accuracy is an upper bound.",
                "The external scenario set has 14 items. Its interval is correspondingly wide.",
                "14 labels only. Anything outside them is answered as the nearest of the 14, "
                "which is why the API ships a confidence floor and an abstention path.",
            ],
        },
        "selected_model": {
            "family": best["model"],
            "classifier_params": best["classifier_params"],
            "vectorizer": best["vectorizer"],
            "validation_accuracy": round(best["accuracy"], 3),
            "validation_macro_f1": round(best["macro_f1"], 3),
            "candidates_evaluated": len(selection_results),
        },
        "cross_validation": {
            "folds": CV_FOLDS,
            "accuracy_mean": round(float(np.mean(cv_accuracies)), 3),
            "accuracy_ci": [round(cv_accuracy_ci[0], 3), round(cv_accuracy_ci[1], 3)],
            "accuracy_per_fold": [round(value, 3) for value in cv_accuracies],
            "macro_f1_mean": round(float(np.mean(cv_macro_f1s)), 3),
            "macro_f1_ci": [round(cv_macro_f1_ci[0], 3), round(cv_macro_f1_ci[1], 3)],
        },
        "test": {
            "rows": len(test),
            "accuracy": round(test_accuracy, 3),
            "accuracy_ci": [round(accuracy_low, 3), round(accuracy_high, 3)],
            "macro_f1": round(test_macro_f1, 3),
            "per_class": per_class,
        },
        "external_use_cases": external,
        "cross_source_transfer": transfer,
        "confidence_floor_sweep": {
            "measured_on": (
                "validation split, scored by a model fit on the training split only, so the "
                "floor is not chosen against data the scoring model has seen"
            ),
            "used_by": "MODEL_CONFIDENCE_FLOOR in app/services/disease_prediction_service.py",
            "rows": floor_sweep,
        },
    }

    # 6. Artifacts.
    CLASSICAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    family_rows = []
    for name, item in sorted(best_by_family.items(), key=lambda pair: pair[1]["macro_f1"], reverse=True):
        family_rows.append(
            {
                "model": name,
                "accuracy": item["accuracy"],
                "macro_f1": item["macro_f1"],
                **calculate_multiclass_outcomes(validation["disease"], item["predictions"], labels),
            }
        )
    save_model_comparison_plot(family_rows, MODEL_COMPARISON_PATH)
    save_prediction_outcomes_plot(family_rows, PREDICTION_OUTCOMES_PATH)
    save_overall_metrics_plot(evaluation, OVERALL_METRICS_PATH)

    joblib.dump(final_pipeline, CLASSICAL_MODEL_PATH)
    joblib.dump(final_pipeline.named_steps["tfidf"], CLASSICAL_VECTORIZER_PATH)

    provenance_counts = Counter(dataset["provenance"])
    metadata = {
        "best_model": best["model"],
        "accuracy": round(test_accuracy, 3),
        "macro_f1": round(test_macro_f1, 3),
        "accuracy_ci_95": [round(accuracy_low, 3), round(accuracy_high, 3)],
        "metric_basis": "held-out test set, never seen by the shipped artifact",
        "best_params": {"classifier": best["classifier_params"], "vectorizer": best["vectorizer"]},
        "dataset_rows": int(len(dataset)),
        "dataset_provenance": {key: int(value) for key, value in sorted(provenance_counts.items())},
        "label_count": int(dataset["disease"].nunique()),
        "labels": labels,
        "evaluation": evaluation,
    }
    CLASSICAL_METADATA_PATH.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    report_lines = [
        "Classical symptom classifier - evaluation report",
        "=" * 55,
        "",
        "METHODOLOGY",
        f"  Split          : group-aware train/validation/test on group_id",
        f"                   train={len(train)} validation={len(validation)} test={len(test)} rows",
        f"  Selection      : {len(selection_results)} pipelines, ranked on VALIDATION macro F1 only",
        f"  Shipped model  : fit on train+validation; the test set never reached it",
        f"  Cross-val      : StratifiedGroupKFold({CV_FOLDS}) over train+validation",
        f"  Intervals      : Wilson (proportions), t across folds (cross-validation)",
        "",
        "HEADLINE (held-out test set)",
        f"  Accuracy       : {test_accuracy:.3f}  95% CI [{accuracy_low:.3f}, {accuracy_high:.3f}]",
        f"  Macro F1       : {test_macro_f1:.3f}",
        "",
        "CROSS-VALIDATION (train+validation)",
        f"  Accuracy       : {np.mean(cv_accuracies):.3f}  95% CI [{cv_accuracy_ci[0]:.3f}, {cv_accuracy_ci[1]:.3f}]",
        f"  Macro F1       : {np.mean(cv_macro_f1s):.3f}  95% CI [{cv_macro_f1_ci[0]:.3f}, {cv_macro_f1_ci[1]:.3f}]",
        f"  Per fold       : {[round(value, 3) for value in cv_accuracies]}",
        "",
        "EXTERNAL SCENARIOS (hand-written, never in the dataset)",
        f"  Accuracy       : {external['accuracy']:.3f} ({external['correct']}/{external['count']})"
        f"  95% CI [{external['accuracy_ci'][0]:.3f}, {external['accuracy_ci'][1]:.3f}]",
    ]
    for failure in external["failures"]:
        report_lines.append(f"    MISS expected={failure['expected']} got={failure['predicted']}")
        report_lines.append(f"         {failure['text']}")

    report_lines += [
        "",
        "CROSS-SOURCE TRANSFER (train without one source passage, test on it)",
        f"  Pooled recall  : {transfer['pooled_recall']} over {transfer['rows_evaluated']} rows",
    ]
    for label in labels:
        entry = transfer["per_label"].get(label, {})
        if entry.get("evaluable"):
            report_lines.append(f"    {label:22s} {entry['mean_recall_on_held_out_passage']}")
        else:
            report_lines.append(f"    {label:22s} not evaluable ({entry.get('reason', 'n/a')})")

    report_lines += [
        "",
        "ENSEMBLE CONFIDENCE FLOOR SWEEP (validation, train-only model)",
        f"    {'floor':>6s} {'accuracy':>9s} {'macro_f1':>9s} {'rule_used':>10s} {'abstain':>8s}",
    ]
    for row in floor_sweep:
        report_lines.append(
            f"    {row['floor']:6.2f} {row['accuracy']:9.3f} {row['macro_f1']:9.3f} "
            f"{row['decided_by_rule_engine']:10d} {row['abstentions']:8d}"
        )

    report_lines += [
        "",
        "PER-CLASS ON TEST SET (recall with 95% Wilson interval)",
        f"    {'label':22s} {'n':>3s}  {'prec':>5s} {'rec':>5s} {'f1':>5s}  recall 95% CI",
    ]
    for entry in per_class:
        report_lines.append(
            f"    {entry['label']:22s} {entry['support']:3d}  {entry['precision']:.3f} "
            f"{entry['recall']:.3f} {entry['f1']:.3f}  [{entry['recall_ci_95'][0]:.3f}, {entry['recall_ci_95'][1]:.3f}]"
        )

    report_lines += [
        "",
        "SELECTED MODEL",
        f"  Family         : {best['model']}",
        f"  Classifier     : {best['classifier_params']}",
        f"  Vectorizer     : {best['vectorizer']}",
        f"  Validation     : accuracy={best['accuracy']:.3f} macro_f1={best['macro_f1']:.3f}",
        "",
        "BEST PER FAMILY (validation set)",
    ]
    for row in family_rows:
        report_lines.append(f"  - {row['model']:24s} accuracy={row['accuracy']:.3f} macro_f1={row['macro_f1']:.3f}")

    report_lines += [
        "",
        "LIMITATIONS",
    ] + [f"  - {item}" for item in evaluation["methodology"]["known_limitations"]] + [
        "",
        "FULL CLASSIFICATION REPORT (test set)",
        classification_report(test["disease"], test_predictions, labels=labels, zero_division=0),
    ]

    CLASSICAL_METRICS_PATH.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print("\n".join(report_lines[:24]))
    print(f"\nSaved model      -> {CLASSICAL_MODEL_PATH}")
    print(f"Saved metrics    -> {CLASSICAL_METRICS_PATH}")
    print(f"Saved metadata   -> {CLASSICAL_METADATA_PATH}")


if __name__ == "__main__":
    main()
