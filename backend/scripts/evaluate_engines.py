"""Measure every prediction engine on one shared held-out split.

This script exists so that the API can never report one engine's score for
another engine's answer. It runs the classical model, the sequence model, the
rule engine and the full ensemble policy over the *same* test rows and the
*same* external scenarios, and writes one file:

    models/evaluation/engine_metrics.json

``DiseasePredictionService.engine_info`` reads an engine's numbers from there.
An engine that has not been measured reports no metrics at all rather than
borrowing someone else's.

It also sweeps the ensemble's confidence floor on the *validation* split and
prints the result, so ``MODEL_CONFIDENCE_FLOOR`` is a choice somebody can check
rather than a number that appeared one day.

    python backend/scripts/evaluate_engines.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BACKEND_DIR))

from app.core.evaluation import load_dataset, three_way_group_split, wilson_interval  # noqa: E402
from app.core.paths import CLASSICAL_METADATA_PATH, ENGINE_METRICS_PATH, EVALUATION_DIR  # noqa: E402
from app.core.use_cases import load_use_cases  # noqa: E402
from app.utils.text_cleaning import clean_text  # noqa: E402
from app.services.disease_prediction_service import (  # noqa: E402
    MODEL_CONFIDENCE_FLOOR,
    DiseasePredictionService,
)
from app.services.prediction_engines import (  # noqa: E402
    CLASSICAL_ENGINE,
    SEQUENCE_ENGINE,
    SYMPTOM_PROFILE_ENGINE,
    SymptomProfileEngine,
)

def score(actual: list[str], predicted: list[str]) -> dict:
    correct = sum(1 for expected, got in zip(actual, predicted) if expected == got)
    low, high = wilson_interval(correct, len(actual))
    return {
        "accuracy": round(accuracy_score(actual, predicted), 3),
        "accuracy_ci_95": [round(low, 3), round(high, 3)],
        "macro_f1": round(f1_score(actual, predicted, average="macro", zero_division=0), 3),
        "correct": correct,
        "rows": len(actual),
    }


def engine_predictions(service: DiseasePredictionService, engine_key: str, frame: pd.DataFrame) -> list[str]:
    """Raw predictions from one engine, with no arbitration involved."""
    engine = service.engines[engine_key]
    predictions: list[str] = []
    for raw_text in frame["text"]:
        outcome = engine.predict(clean_text(str(raw_text)))
        # An engine with no opinion is scored as wrong rather than skipped:
        # silence is a real outcome and hiding it would flatter the engine.
        predictions.append(outcome.disease or "no_opinion")
    return predictions


def ensemble_predictions(service: DiseasePredictionService, frame: pd.DataFrame, model_key: str) -> list[str]:
    return [service.predict(str(text), model_key=model_key).predicted_disease for text in frame["text"]]


def main() -> None:
    dataset = load_dataset()
    split = three_way_group_split(dataset)
    labels = sorted(dataset["disease"].unique().tolist())

    # Rule-engine profiles are built from train+validation for measurement. At
    # runtime it legitimately uses the whole dataset -- there is no test set in
    # production -- but scoring it against rows that shaped its own profiles
    # would be measuring nothing.
    service = DiseasePredictionService()
    service.engines[SYMPTOM_PROFILE_ENGINE] = SymptomProfileEngine(profile_source=split.development)

    use_cases = load_use_cases(labels=set(labels))
    external_frame = pd.DataFrame(
        {"text": [case.text for case in use_cases], "disease": [case.expected for case in use_cases]}
    )

    print(f"test rows={len(split.test)}  external scenarios={len(external_frame)}")

    classical_metadata = json.loads(CLASSICAL_METADATA_PATH.read_text(encoding="utf-8"))
    transfer = (
        classical_metadata.get("evaluation", {}).get("cross_source_transfer", {}).get("pooled_recall")
    )

    engines: dict[str, dict] = {}
    for engine_key in (CLASSICAL_ENGINE, SEQUENCE_ENGINE, SYMPTOM_PROFILE_ENGINE):
        engine = service.engines[engine_key]
        if not engine.is_available():
            engines[engine_key] = {
                "available": False,
                "basis": "engine not available in this environment; nothing measured",
            }
            print(f"  {engine_key:18s} unavailable")
            continue

        test_scores = score(list(split.test["disease"]), engine_predictions(service, engine_key, split.test))
        external_scores = (
            score(list(external_frame["disease"]), engine_predictions(service, engine_key, external_frame))
            if len(external_frame)
            else {}
        )

        entry = {
            "available": True,
            **{key: value for key, value in test_scores.items() if key != "correct"},
            "external_use_case_accuracy": external_scores.get("accuracy"),
            "external_use_case_rows": external_scores.get("rows"),
            "basis": (
                f"held-out test set of {len(split.test)} rows, group-aware split, "
                f"engine run standalone with no arbitration"
            ),
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
        }
        if engine_key == CLASSICAL_ENGINE and transfer is not None:
            entry["cross_source_transfer_recall"] = transfer
        engines[engine_key] = entry
        print(
            f"  {engine_key:18s} test accuracy={entry['accuracy']:.3f} "
            f"macro_f1={entry['macro_f1']:.3f} external={entry['external_use_case_accuracy']}"
        )

    ensemble_test = score(list(split.test["disease"]), ensemble_predictions(service, split.test, CLASSICAL_ENGINE))
    ensemble_external = (
        score(list(external_frame["disease"]), ensemble_predictions(service, external_frame, CLASSICAL_ENGINE))
        if len(external_frame)
        else {}
    )
    print(f"  {'ensemble':18s} test accuracy={ensemble_test['accuracy']:.3f}")

    document = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "methodology": (
            "Every engine is run standalone over the same group-aware held-out test split "
            "produced by app.core.evaluation.three_way_group_split, and over the hand-written "
            "scenarios in tests/use_case_tests.txt. An engine that declines to answer is scored "
            "as wrong, not skipped. The ensemble row is the full arbitration policy end to end."
        ),
        "split": {
            "train_rows": len(split.train),
            "validation_rows": len(split.validation),
            "test_rows": len(split.test),
        },
        "engines": engines,
        "ensemble": {
            **{key: value for key, value in ensemble_test.items() if key != "correct"},
            "external_use_case_accuracy": ensemble_external.get("accuracy"),
            "policy_confidence_floor": MODEL_CONFIDENCE_FLOOR,
            "basis": "full arbitration policy over the held-out test set",
        },
        "confidence_floor_note": (
            "The sweep that chooses MODEL_CONFIDENCE_FLOOR lives in train_model.py and is "
            "recorded under evaluation.confidence_floor_sweep in "
            "models/classical/model_metadata.json. It cannot run here: the shipped artifact "
            "is fit on train+validation, so scoring it against validation returns its own "
            "training data and every floor looks perfect."
        ),
    }

    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    ENGINE_METRICS_PATH.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved -> {ENGINE_METRICS_PATH}")


if __name__ == "__main__":
    main()
