"""Optional sequence-model experiment (LSTM / GRU), evaluated the same way.

What was wrong with the previous version
----------------------------------------
``fit()`` was called with ``validation_data=(x_test, y_test)`` while
``EarlyStopping(restore_best_weights=True)`` watched ``val_accuracy``. The test
set was therefore choosing the model's weights: whichever epoch happened to
score best *on the test set* is the epoch that got restored and then reported.
The resulting 0.844 was not an estimate of anything.

This version uses the same protocol as the classical pipeline: a group-aware
train/validation/test split, early stopping and architecture choice driven by
the validation set alone, and the test set read exactly once at the end.

It remains optional. TensorFlow is a large dependency, the classical model wins
on this dataset, and the API works without any of this.

    pip install -r backend/requirements-advanced.txt
    python backend/scripts/train_deep_learning_model.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.metrics import accuracy_score, classification_report, f1_score  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402
from sklearn.preprocessing import LabelEncoder  # noqa: E402

try:
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import (
        Bidirectional,
        Dense,
        Dropout,
        Embedding,
        GlobalMaxPooling1D,
        GRU,
        Input,
        LSTM,
    )
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.utils import to_categorical
    import tensorflow as tf
except ImportError as exc:  # pragma: no cover - exercised only without TensorFlow
    raise SystemExit(
        "TensorFlow is required for the optional advanced model. "
        "Install it with: pip install -r backend/requirements-advanced.txt"
    ) from exc

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BACKEND_DIR))

from app.core.paths import (  # noqa: E402
    ADVANCED_LABEL_ENCODER_PATH,
    ADVANCED_METADATA_PATH,
    ADVANCED_METRICS_PATH,
    ADVANCED_MODEL_DIR,
    ADVANCED_MODEL_PATH,
    ADVANCED_OVERALL_METRICS_PATH,
    ADVANCED_TOKENIZER_PATH,
    ADVANCED_TRAINING_HISTORY_PATH,
    DATASET_PATH,
)
from app.core.use_cases import load_use_cases  # noqa: E402
from app.utils.text_cleaning import clean_text  # noqa: E402

MAX_WORDS = 4000
MAX_SEQUENCE_LENGTH = 48
EPOCHS = 60
RANDOM_STATE = 42

# A small, documented architecture sweep. Every candidate is scored on the
# validation set; the test set is not involved in this choice.
CANDIDATES: tuple[dict, ...] = (
    {"name": "bilstm_64", "model_type": "lstm", "units": 64, "embedding_dim": 96, "dropout": 0.3, "learning_rate": 1e-3, "batch_size": 16},
    {"name": "bilstm_32", "model_type": "lstm", "units": 32, "embedding_dim": 64, "dropout": 0.3, "learning_rate": 1e-3, "batch_size": 16},
    {"name": "bigru_64", "model_type": "gru", "units": 64, "embedding_dim": 96, "dropout": 0.3, "learning_rate": 1e-3, "batch_size": 16},
    {"name": "bigru_32", "model_type": "gru", "units": 32, "embedding_dim": 64, "dropout": 0.2, "learning_rate": 7e-4, "batch_size": 8},
)


def build_model(config: dict, class_count: int) -> Model:
    inputs = Input(shape=(MAX_SEQUENCE_LENGTH,))
    hidden = Embedding(input_dim=MAX_WORDS, output_dim=config["embedding_dim"])(inputs)

    recurrent = LSTM if config["model_type"] == "lstm" else GRU
    hidden = Bidirectional(recurrent(config["units"], return_sequences=True))(hidden)
    hidden = GlobalMaxPooling1D()(hidden)
    hidden = Dropout(config["dropout"])(hidden)
    hidden = Dense(64, activation="relu")(hidden)
    hidden = Dropout(config["dropout"])(hidden)
    outputs = Dense(class_count, activation="softmax")(hidden)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=config["learning_rate"]),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def group_holdout(frame: pd.DataFrame, n_folds: int, random_state: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    remainder_index, holdout_index = next(
        splitter.split(frame["cleaned_text"], frame["disease"], groups=frame["group_id"])
    )
    return frame.iloc[remainder_index].copy(), frame.iloc[holdout_index].copy()


def encode(tokenizer: Tokenizer, texts) -> np.ndarray:
    return pad_sequences(
        tokenizer.texts_to_sequences(texts),
        maxlen=MAX_SEQUENCE_LENGTH,
        padding="post",
        truncating="post",
    )


def save_history_plot(history: dict, output_path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(history["accuracy"], label="train")
    axes[0].plot(history["val_accuracy"], label="validation")
    axes[0].set_title("Accuracy per epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(history["loss"], label="train")
    axes[1].plot(history["val_loss"], label="validation")
    axes[1].set_title("Loss per epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    figure.suptitle("Selected sequence model (validation curve drives early stopping)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_overall_plot(validation_score: float, test_accuracy: float, test_macro_f1: float, output_path: Path) -> None:
    names = ["Validation accuracy", "Test accuracy", "Test macro F1"]
    values = [validation_score, test_accuracy, test_macro_f1]
    plt.figure(figsize=(7.5, 5))
    axis = sns.barplot(x=names, y=values, hue=names, palette="Purples", legend=False)
    axis.set_ylim(0, 1.05)
    axis.set_title("Sequence model - selection vs held-out test")
    for index, value in enumerate(values):
        axis.text(index, value + 0.02, f"{value:.3f}", ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    tf.keras.utils.set_random_seed(RANDOM_STATE)

    dataset = pd.read_csv(DATASET_PATH)
    dataset["cleaned_text"] = dataset["text"].apply(clean_text)
    labels = sorted(dataset["disease"].unique().tolist())

    development, test = group_holdout(dataset, 5, RANDOM_STATE)
    train, validation = group_holdout(development, 4, RANDOM_STATE)

    # The tokenizer is fit on training text only. Fitting it on everything would
    # leak test vocabulary into the model's input representation.
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<oov>")
    tokenizer.fit_on_texts(train["cleaned_text"])

    encoder = LabelEncoder().fit(labels)
    x_train, x_validation, x_test = (encode(tokenizer, frame["cleaned_text"]) for frame in (train, validation, test))
    y_train = to_categorical(encoder.transform(train["disease"]), num_classes=len(labels))
    y_validation = to_categorical(encoder.transform(validation["disease"]), num_classes=len(labels))

    print(f"rows: train={len(train)} validation={len(validation)} test={len(test)}")

    results: list[dict] = []
    for config in CANDIDATES:
        model = build_model(config, len(labels))
        early_stopping = EarlyStopping(
            monitor="val_accuracy", mode="max", patience=10, restore_best_weights=True
        )
        history = model.fit(
            x_train,
            y_train,
            # Validation only. The test set is not visible to fit() or to early
            # stopping -- that was the bug this rewrite exists to remove.
            validation_data=(x_validation, y_validation),
            epochs=EPOCHS,
            batch_size=config["batch_size"],
            callbacks=[early_stopping],
            verbose=0,
        )
        validation_predictions = np.argmax(model.predict(x_validation, verbose=0), axis=1)
        validation_actual = encoder.transform(validation["disease"])
        score = float(f1_score(validation_actual, validation_predictions, average="macro", zero_division=0))
        accuracy = float(accuracy_score(validation_actual, validation_predictions))
        print(f"  {config['name']:12s} validation accuracy={accuracy:.3f} macro_f1={score:.3f}")
        results.append({"config": config, "model": model, "history": history.history, "macro_f1": score, "accuracy": accuracy})

    best = max(results, key=lambda item: (item["macro_f1"], item["accuracy"]))
    print(f"selected on validation: {best['config']['name']}")

    # The test set is opened once, here, by one model.
    test_predictions = np.argmax(best["model"].predict(x_test, verbose=0), axis=1)
    test_actual = encoder.transform(test["disease"])
    test_accuracy = float(accuracy_score(test_actual, test_predictions))
    test_macro_f1 = float(f1_score(test_actual, test_predictions, average="macro", zero_division=0))

    use_cases = load_use_cases(labels=set(labels))
    external_correct = 0
    if use_cases:
        external_input = encode(tokenizer, [clean_text(case.text) for case in use_cases])
        external_predictions = encoder.inverse_transform(
            np.argmax(best["model"].predict(external_input, verbose=0), axis=1)
        )
        external_correct = sum(
            1 for case, prediction in zip(use_cases, external_predictions) if prediction == case.expected
        )

    ADVANCED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    best["model"].save(ADVANCED_MODEL_PATH)
    joblib.dump(tokenizer, ADVANCED_TOKENIZER_PATH)
    joblib.dump(encoder, ADVANCED_LABEL_ENCODER_PATH)
    save_history_plot(best["history"], ADVANCED_TRAINING_HISTORY_PATH)
    save_overall_plot(best["accuracy"], test_accuracy, test_macro_f1, ADVANCED_OVERALL_METRICS_PATH)

    metadata = {
        "purpose": "Optional sequence-model (LSTM/GRU) comparison against the classical pipeline.",
        "best_model": best["config"]["name"],
        "best_model_type": best["config"]["model_type"],
        "best_config": best["config"],
        "accuracy": round(test_accuracy, 3),
        "macro_f1": round(test_macro_f1, 3),
        "metric_basis": "held-out test set, opened once after selection on validation",
        "dataset_rows": int(len(dataset)),
        "label_count": len(labels),
        "labels": labels,
        "max_words": MAX_WORDS,
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "evaluation": {
            "split": "group-aware train/validation/test on dataset group_id",
            "split_sizes": {"train": len(train), "validation": len(validation), "test": len(test)},
            "selection": f"{len(CANDIDATES)} architectures, ranked on validation macro F1",
            "early_stopping": "monitors validation accuracy; the test set is never passed to fit()",
            "tokenizer_fit_on": "training split only",
            "validation_accuracy": round(best["accuracy"], 3),
            "validation_macro_f1": round(best["macro_f1"], 3),
            "external_use_cases": {
                "count": len(use_cases),
                "correct": external_correct,
                "accuracy": round(external_correct / len(use_cases), 3) if use_cases else None,
            },
            "candidates": [
                {"name": item["config"]["name"], "validation_accuracy": round(item["accuracy"], 3),
                 "validation_macro_f1": round(item["macro_f1"], 3)}
                for item in results
            ],
        },
        "note": (
            "Artifact paths are intentionally omitted: the previous metadata recorded absolute "
            "paths from one author's machine. Everything lives beside this file in models/advanced/."
        ),
    }
    ADVANCED_METADATA_PATH.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    report = [
        "Advanced sequence model - evaluation report",
        "=" * 55,
        "",
        "METHODOLOGY",
        "  Split        : group-aware train/validation/test on group_id",
        f"                 train={len(train)} validation={len(validation)} test={len(test)}",
        f"  Selection    : {len(CANDIDATES)} architectures ranked on VALIDATION macro F1",
        "  Early stop   : monitors validation accuracy; test set never passed to fit()",
        "  Tokenizer    : fit on the training split only",
        "",
        "RESULTS",
        f"  Selected     : {best['config']['name']}",
        f"  Validation   : accuracy={best['accuracy']:.3f} macro_f1={best['macro_f1']:.3f}",
        f"  Test         : accuracy={test_accuracy:.3f} macro_f1={test_macro_f1:.3f}",
        f"  External     : {external_correct}/{len(use_cases)} hand-written scenarios",
        "",
        "CANDIDATES (validation)",
    ]
    for item in results:
        report.append(
            f"  - {item['config']['name']:12s} accuracy={item['accuracy']:.3f} macro_f1={item['macro_f1']:.3f}"
        )
    report += [
        "",
        "TEST CLASSIFICATION REPORT",
        classification_report(
            encoder.inverse_transform(test_actual),
            encoder.inverse_transform(test_predictions),
            labels=labels,
            zero_division=0,
        ),
    ]
    ADVANCED_METRICS_PATH.write_text("\n".join(report) + "\n", encoding="utf-8")

    print("\n".join(report[:18]))
    print(f"\nSaved model    -> {ADVANCED_MODEL_PATH}")
    print(f"Saved metadata -> {ADVANCED_METADATA_PATH}")


if __name__ == "__main__":
    main()
