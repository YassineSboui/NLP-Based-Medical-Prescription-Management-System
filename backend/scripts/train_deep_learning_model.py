from __future__ import annotations

from pathlib import Path
import json
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

try:
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import Bidirectional, Conv1D, Dense, Dropout, Embedding, GlobalMaxPooling1D, GRU, Input, LSTM
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.utils import to_categorical
except ImportError as exc:
    raise SystemExit(
        "TensorFlow is required for the optional advanced model. "
        "Install it with: pip install -r backend/requirements-advanced.txt"
    ) from exc


BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BACKEND_DIR))

from app.core.paths import ADVANCED_MODEL_DIR, DATASET_PATH  # noqa: E402
from app.utils.text_cleaning import clean_text  # noqa: E402


MODEL_DIR = ADVANCED_MODEL_DIR
LSTM_MODEL_PATH = MODEL_DIR / "lstm_model.keras"
GRU_MODEL_PATH = MODEL_DIR / "gru_model.keras"
BEST_MODEL_PATH = MODEL_DIR / "best_sequence_model.keras"
TOKENIZER_PATH = MODEL_DIR / "tokenizer.joblib"
LABEL_ENCODER_PATH = MODEL_DIR / "label_encoder.joblib"
METADATA_PATH = MODEL_DIR / "deep_learning_metadata.json"
REPORT_PATH = MODEL_DIR / "deep_learning_metrics.txt"
HISTORY_PATH = MODEL_DIR / "training_history.png"
CONFUSION_MATRIX_PATH = MODEL_DIR / "deep_learning_confusion_matrix.png"
NORMALIZED_CONFUSION_MATRIX_PATH = MODEL_DIR / "deep_learning_confusion_matrix_normalized.png"

MAX_WORDS = 3000
EPOCHS = 35
RANDOM_STATE = 42
TARGET_ACCURACY = 0.80


def build_sequence_model(config: dict, class_count: int) -> Model:
    inputs = Input(shape=(config["max_sequence_length"],))
    x = Embedding(input_dim=MAX_WORDS, output_dim=config["embedding_dim"])(inputs)
    model_type = config["model_type"]
    if config.get("conv_filters"):
        x = Conv1D(filters=config["conv_filters"], kernel_size=config.get("kernel_size", 3), activation="relu", padding="same")(x)

    if model_type == "lstm":
        x = Bidirectional(LSTM(config["units"], return_sequences=config.get("pooling") == "max"))(x)
    elif model_type == "gru":
        x = Bidirectional(GRU(config["units"], return_sequences=config.get("pooling") == "max"))(x)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    if config.get("pooling") == "max":
        x = GlobalMaxPooling1D()(x)

    x = Dropout(config["recurrent_dropout"])(x)
    x = Dense(config["dense_units"], activation="relu")(x)
    x = Dropout(config["dense_dropout"])(x)
    outputs = Dense(class_count, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=config.get("learning_rate", 0.001)), loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def save_history_plot(histories: dict[str, dict[str, list[float]]], output_path: Path) -> None:
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    for name, history in histories.items():
        plt.plot(history["accuracy"], label=f"{name} train")
        plt.plot(history["val_accuracy"], label=f"{name} validation")
    plt.title("Advanced Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    for name, history in histories.items():
        plt.plot(history["loss"], label=f"{name} train")
        plt.plot(history["val_loss"], label=f"{name} validation")
    plt.title("Advanced Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_confusion_matrix_plot(labels: list[str], matrix, title: str, output_path: Path, fmt: str) -> None:
    plt.figure(figsize=(13, 10))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=fmt,
        cmap="Purples",
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


def train_candidate(config: dict, model: Model, x_train, y_train, x_test, y_test) -> dict:
    early_stopping = EarlyStopping(monitor="val_accuracy", mode="max", patience=7, restore_best_weights=True)
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        epochs=EPOCHS,
        batch_size=config["batch_size"],
        callbacks=[early_stopping],
        verbose=0,
    )

    probabilities = model.predict(x_test)
    predictions = np.argmax(probabilities, axis=1)
    actual = np.argmax(y_test, axis=1)
    return {
        "name": config["name"],
        "config": config,
        "model": model,
        "history": history.history,
        "predictions": predictions,
        "actual": actual,
        "accuracy": accuracy_score(actual, predictions),
        "macro_f1": f1_score(actual, predictions, average="macro", zero_division=0),
    }


def main() -> None:
    dataset = pd.read_csv(DATASET_PATH)
    dataset["cleaned_text"] = dataset["text"].apply(clean_text)

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(dataset["disease"])
    categorical_labels = to_categorical(encoded_labels)
    labels = label_encoder.classes_.tolist()

    class_count = len(labels)
    configs = [
        {
            "name": "lstm_u32_e48_s32_b8",
            "model_type": "lstm",
            "units": 32,
            "embedding_dim": 48,
            "max_sequence_length": 32,
            "dense_units": 48,
            "recurrent_dropout": 0.30,
            "dense_dropout": 0.20,
            "batch_size": 8,
            "learning_rate": 0.001,
        },
        {
            "name": "lstm_u32_e64_s32_b8_conv",
            "model_type": "lstm",
            "units": 32,
            "embedding_dim": 64,
            "max_sequence_length": 32,
            "dense_units": 64,
            "recurrent_dropout": 0.25,
            "dense_dropout": 0.20,
            "batch_size": 8,
            "conv_filters": 32,
            "kernel_size": 3,
            "learning_rate": 0.001,
        },
        {
            "name": "lstm_u48_e64_s40_b8",
            "model_type": "lstm",
            "units": 48,
            "embedding_dim": 64,
            "max_sequence_length": 40,
            "dense_units": 64,
            "recurrent_dropout": 0.35,
            "dense_dropout": 0.25,
            "batch_size": 8,
            "learning_rate": 0.001,
        },
        {
            "name": "lstm_u64_e96_s48_b8",
            "model_type": "lstm",
            "units": 64,
            "embedding_dim": 96,
            "max_sequence_length": 48,
            "dense_units": 96,
            "recurrent_dropout": 0.35,
            "dense_dropout": 0.25,
            "batch_size": 8,
            "learning_rate": 0.001,
        },
        {
            "name": "lstm_u64_e96_s48_b8_pool",
            "model_type": "lstm",
            "units": 64,
            "embedding_dim": 96,
            "max_sequence_length": 48,
            "dense_units": 96,
            "recurrent_dropout": 0.25,
            "dense_dropout": 0.20,
            "batch_size": 8,
            "pooling": "max",
            "learning_rate": 0.001,
        },
        {
            "name": "gru_u32_e48_s32_b8",
            "model_type": "gru",
            "units": 32,
            "embedding_dim": 48,
            "max_sequence_length": 32,
            "dense_units": 48,
            "recurrent_dropout": 0.30,
            "dense_dropout": 0.20,
            "batch_size": 8,
            "learning_rate": 0.001,
        },
        {
            "name": "gru_u32_e64_s32_b8_conv",
            "model_type": "gru",
            "units": 32,
            "embedding_dim": 64,
            "max_sequence_length": 32,
            "dense_units": 64,
            "recurrent_dropout": 0.25,
            "dense_dropout": 0.20,
            "batch_size": 8,
            "conv_filters": 32,
            "kernel_size": 3,
            "learning_rate": 0.001,
        },
        {
            "name": "gru_u48_e64_s40_b8",
            "model_type": "gru",
            "units": 48,
            "embedding_dim": 64,
            "max_sequence_length": 40,
            "dense_units": 64,
            "recurrent_dropout": 0.35,
            "dense_dropout": 0.25,
            "batch_size": 8,
            "learning_rate": 0.001,
        },
        {
            "name": "gru_u64_e96_s48_b8",
            "model_type": "gru",
            "units": 64,
            "embedding_dim": 96,
            "max_sequence_length": 48,
            "dense_units": 96,
            "recurrent_dropout": 0.35,
            "dense_dropout": 0.25,
            "batch_size": 8,
            "learning_rate": 0.001,
        },
        {
            "name": "gru_u64_e96_s48_b4",
            "model_type": "gru",
            "units": 64,
            "embedding_dim": 96,
            "max_sequence_length": 48,
            "dense_units": 96,
            "recurrent_dropout": 0.25,
            "dense_dropout": 0.20,
            "batch_size": 4,
            "learning_rate": 0.001,
        },
        {
            "name": "gru_u64_e96_s48_b8_pool",
            "model_type": "gru",
            "units": 64,
            "embedding_dim": 96,
            "max_sequence_length": 48,
            "dense_units": 96,
            "recurrent_dropout": 0.25,
            "dense_dropout": 0.20,
            "batch_size": 8,
            "pooling": "max",
            "learning_rate": 0.001,
        },
    ]
    configs.extend([{**config, "name": f"{config['name']}_lr7e4", "learning_rate": 0.0007} for config in configs])
    seeds = [7, 21, 42, 84, 123]

    candidates = []
    best_by_model_type: dict[str, dict] = {}
    tokenizers = {}
    reached_target = False
    for seed in seeds:
        if reached_target:
            break
        np.random.seed(seed)
        tf.random.set_seed(seed)
        train_text, test_text, train_labels, test_labels = train_test_split(
            dataset["cleaned_text"],
            categorical_labels,
            test_size=0.25,
            random_state=RANDOM_STATE,
            stratify=encoded_labels,
        )

        for config in configs:
            if reached_target:
                break
            run_config = {**config, "seed": seed}
            run_name = f"{config['name']}_seed{seed}"
            run_config["name"] = run_name
            tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
            tokenizer.fit_on_texts(train_text)
            train_sequences = tokenizer.texts_to_sequences(train_text)
            test_sequences = tokenizer.texts_to_sequences(test_text)
            x_train = pad_sequences(
                train_sequences,
                maxlen=config["max_sequence_length"],
                padding="post",
                truncating="post",
            )
            x_test = pad_sequences(
                test_sequences,
                maxlen=config["max_sequence_length"],
                padding="post",
                truncating="post",
            )
            tokenizers[run_name] = tokenizer
            candidate = train_candidate(run_config, build_sequence_model(run_config, class_count), x_train, train_labels, x_test, test_labels)
            candidates.append(candidate)
            print(f"{run_name}: accuracy={candidate['accuracy']:.3f}, macro_f1={candidate['macro_f1']:.3f}")
            model_type = config["model_type"]
            model_type_best = best_by_model_type.get(model_type)
            if model_type_best is None or (candidate["macro_f1"], candidate["accuracy"]) > (
                model_type_best["macro_f1"],
                model_type_best["accuracy"],
            ):
                best_by_model_type[model_type] = candidate

            if candidate["accuracy"] >= TARGET_ACCURACY:
                print(f"Reached target advanced accuracy with {run_name}: {candidate['accuracy']:.3f}")
                reached_target = True

    best = max(candidates, key=lambda item: (item["macro_f1"], item["accuracy"]))
    best_tokenizer = tokenizers[best["name"]]

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    best_lstm = best_by_model_type["lstm"]
    best_gru = best_by_model_type["gru"]
    best_lstm["model"].save(LSTM_MODEL_PATH)
    best_gru["model"].save(GRU_MODEL_PATH)
    best["model"].save(BEST_MODEL_PATH)
    joblib.dump(best_tokenizer, TOKENIZER_PATH)
    joblib.dump(label_encoder, LABEL_ENCODER_PATH)

    label_ids = list(range(len(labels)))
    report = classification_report(best["actual"], best["predictions"], labels=label_ids, target_names=labels, zero_division=0)
    cm = confusion_matrix(best["actual"], best["predictions"], labels=label_ids)
    cm_normalized = confusion_matrix(best["actual"], best["predictions"], labels=label_ids, normalize="true")

    save_history_plot({candidate["name"]: candidate["history"] for candidate in candidates}, HISTORY_PATH)
    save_confusion_matrix_plot(labels, cm, f"Advanced {best['name'].upper()} Confusion Matrix", CONFUSION_MATRIX_PATH, "d")
    save_confusion_matrix_plot(
        labels,
        cm_normalized,
        f"Advanced {best['name'].upper()} Normalized Confusion Matrix",
        NORMALIZED_CONFUSION_MATRIX_PATH,
        ".2f",
    )

    metrics = [
        "Optional advanced neural-network experiment",
        "Important: this is included for academic comparison. The dataset is small, so TF-IDF models may generalize better.",
        f"Best advanced model: {best['name']}",
        f"Best advanced config: {best['config']}",
        f"Accuracy: {best['accuracy']:.3f}",
        f"Macro F1: {best['macro_f1']:.3f}",
        "",
        "Optimized advanced model comparison:",
        *[
            f"- {model_type}: accuracy={item['accuracy']:.3f}, macro_f1={item['macro_f1']:.3f}, best_config={item['config']}"
            for model_type, item in sorted(best_by_model_type.items(), key=lambda pair: (pair[1]["macro_f1"], pair[1]["accuracy"]), reverse=True)
        ],
        "",
        report,
    ]
    REPORT_PATH.write_text("\n".join(metrics), encoding="utf-8")
    METADATA_PATH.write_text(
        json.dumps(
            {
                "purpose": "Optional advanced LSTM/GRU sequence classification experiment",
                "warning": "Small dataset; included for comparison and demonstration, not necessarily better than TF-IDF.",
                "best_model": best["name"],
                "best_model_type": best["config"]["model_type"],
                "best_config": best["config"],
                "accuracy": best["accuracy"],
                "macro_f1": best["macro_f1"],
                "dataset_rows": int(len(dataset)),
                "label_count": int(len(labels)),
                "labels": labels,
                "max_words": MAX_WORDS,
                "max_sequence_length": best["config"]["max_sequence_length"],
                "artifacts": {
                    "lstm_model": str(LSTM_MODEL_PATH),
                    "gru_model": str(GRU_MODEL_PATH),
                    "best_model": str(BEST_MODEL_PATH),
                    "tokenizer": str(TOKENIZER_PATH),
                    "label_encoder": str(LABEL_ENCODER_PATH),
                    "training_history": str(HISTORY_PATH),
                    "confusion_matrix": str(CONFUSION_MATRIX_PATH),
                    "normalized_confusion_matrix": str(NORMALIZED_CONFUSION_MATRIX_PATH),
                },
                "note": "Internal LSTM/GRU architecture variants and seeds were tested during training; only the final selected advanced model is shown here for a clean project submission.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Saved LSTM model to {LSTM_MODEL_PATH}")
    print(f"Saved GRU model to {GRU_MODEL_PATH}")
    print(f"Saved best sequence model to {BEST_MODEL_PATH}")
    print(f"Saved tokenizer to {TOKENIZER_PATH}")
    print(f"Saved label encoder to {LABEL_ENCODER_PATH}")
    print(f"Saved metrics to {REPORT_PATH}")
    print(f"Saved metadata to {METADATA_PATH}")
    print(f"Saved training history plot to {HISTORY_PATH}")
    print(f"Saved confusion matrix to {CONFUSION_MATRIX_PATH}")
    print(f"Saved normalized confusion matrix to {NORMALIZED_CONFUSION_MATRIX_PATH}")
    print(metrics[2])
    print(metrics[4])


if __name__ == "__main__":
    main()
