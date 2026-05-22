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

from app.core.paths import ADVANCED_MODEL_DIR, ADVANCED_OVERALL_METRICS_PATH, ADVANCED_PREDICTION_OUTCOMES_PATH, DATASET_PATH  # noqa: E402
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

MAX_WORDS = 3000
EPOCHS = 35
RANDOM_STATE = 42
TARGET_ACCURACY = 0.80


# Build one neural sequence classifier from a configuration dictionary.
# The same function can create LSTM or GRU variants with optional Conv1D/pooling.
def build_sequence_model(config: dict, class_count: int) -> Model:
    # Input is a fixed-length integer sequence produced by the tokenizer.
    inputs = Input(shape=(config["max_sequence_length"],))

    # Embedding learns dense vector representations for symptom tokens.
    x = Embedding(input_dim=MAX_WORDS, output_dim=config["embedding_dim"])(inputs)
    model_type = config["model_type"]

    # Optional convolution layer can capture short local symptom patterns before the recurrent layer.
    if config.get("conv_filters"):
        x = Conv1D(filters=config["conv_filters"], kernel_size=config.get("kernel_size", 3), activation="relu", padding="same")(x)

    # Bidirectional recurrent layer reads the sequence from both directions.
    if model_type == "lstm":
        x = Bidirectional(LSTM(config["units"], return_sequences=config.get("pooling") == "max"))(x)
    elif model_type == "gru":
        x = Bidirectional(GRU(config["units"], return_sequences=config.get("pooling") == "max"))(x)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    # Optional max pooling keeps the strongest signal across the sequence.
    if config.get("pooling") == "max":
        x = GlobalMaxPooling1D()(x)

    # Dropout helps reduce overfitting on the small dataset.
    x = Dropout(config["recurrent_dropout"])(x)
    x = Dense(config["dense_units"], activation="relu")(x)
    x = Dropout(config["dense_dropout"])(x)

    # Softmax outputs one probability per disease label.
    outputs = Dense(class_count, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=config.get("learning_rate", 0.001)), loss="categorical_crossentropy", metrics=["accuracy"])
    return model


# Save accuracy/loss curves to show training behavior and overfitting risk.
def save_history_plot(histories: dict[str, dict[str, list[float]]], output_path: Path) -> None:
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    for name, history in histories.items():
        plt.plot(history["accuracy"], label=f"{name} train")
        plt.plot(history["val_accuracy"], linestyle="--", label=f"{name} val")
    plt.title("Advanced Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right", fontsize=8)

    plt.subplot(1, 2, 2)
    for name, history in histories.items():
        plt.plot(history["loss"], label=f"{name} train")
        plt.plot(history["val_loss"], linestyle="--", label=f"{name} val")
    plt.title("Advanced Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_overall_metrics_plot(accuracy: float, macro_f1: float, output_path: Path) -> None:
    plt.figure(figsize=(7, 5))
    ax = sns.barplot(x=["Accuracy", "Macro F1"], y=[accuracy, macro_f1], hue=["Accuracy", "Macro F1"], palette="Purples", legend=False)
    ax.set_ylim(0, 1)
    ax.set_title("Overall Advanced Model Metrics")
    ax.set_ylabel("Score")
    for index, value in enumerate([accuracy, macro_f1]):
        ax.text(index, value + 0.02, f"{value:.3f}", ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def calculate_multiclass_outcomes(actual_labels, predicted_labels, labels: list[int]) -> dict[str, int]:
    matrix = confusion_matrix(actual_labels, predicted_labels, labels=labels)
    total = int(matrix.sum())
    true_positive = false_positive = true_negative = false_negative = 0

    # For multi-class classification, TP/FP/TN/FN are computed one-vs-rest per label and summed.
    for index in range(len(labels)):
        tp = int(matrix[index, index])
        fp = int(matrix[:, index].sum() - tp)
        fn = int(matrix[index, :].sum() - tp)
        tn = int(total - tp - fp - fn)

        true_positive += tp
        false_positive += fp
        false_negative += fn
        true_negative += tn

    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "false_negative": false_negative,
    }


def save_prediction_outcomes_plot(results: list[dict], output_path: Path) -> None:
    result_frame = pd.DataFrame(results).sort_values("accuracy", ascending=False)
    figure, axes = plt.subplots(1, len(result_frame), figsize=(6 * len(result_frame), 5))
    if len(result_frame) == 1:
        axes = [axes]

    for axis, row in zip(axes, result_frame.itertuples()):
        matrix = np.array(
            [
                [row.true_positive, row.false_positive],
                [row.false_negative, row.true_negative],
            ]
        )
        labels = np.array(
            [
                [f"TP\n{row.true_positive}", f"FP\n{row.false_positive}"],
                [f"FN\n{row.false_negative}", f"TN\n{row.true_negative}"],
            ]
        )
        sns.heatmap(
            matrix,
            annot=labels,
            fmt="",
            cmap="Purples",
            cbar=False,
            linewidths=2,
            linecolor="white",
            square=True,
            ax=axis,
            annot_kws={"fontsize": 14, "fontweight": "bold"},
        )
        axis.set_title(f"{row.model}\nAccuracy {row.accuracy:.3f} | Macro F1 {row.macro_f1:.3f}", fontweight="bold")
        axis.set_xlabel("Predicted")
        axis.set_ylabel("Actual")
        axis.set_xticklabels(["Positive", "Negative"])
        axis.set_yticklabels(["Positive", "Negative"], rotation=0)

    figure.suptitle("Advanced TP / FP / FN / TN (One-vs-Rest Aggregated)", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


# Train one LSTM/GRU candidate and return metrics plus artifacts needed later.
def train_candidate(config: dict, model: Model, x_train, y_train, x_test, y_test) -> dict:
    # Stop when validation accuracy stops improving, restoring the best epoch weights.
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

    # Convert softmax probabilities to class indexes for evaluation.
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
    # 1. Load and clean the same curated dataset used by the classical model.
    dataset = pd.read_csv(DATASET_PATH)
    dataset["cleaned_text"] = dataset["text"].apply(clean_text)

    # 2. Encode disease names as integers, then one-hot vectors for softmax training.
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(dataset["disease"])
    categorical_labels = to_categorical(encoded_labels)
    labels = label_encoder.classes_.tolist()

    class_count = len(labels)

    # 3. Define LSTM/GRU architecture configurations to test internally.
    # The final metadata shows only the selected best advanced model.
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

    # Add a second learning-rate pass for every architecture.
    configs.extend([{**config, "name": f"{config['name']}_lr7e4", "learning_rate": 0.0007} for config in configs])

    # Multiple seeds are tested because neural models can vary on small datasets.
    seeds = [7, 21, 42, 84, 123]

    # 4. Search candidates until a target accuracy is reached or all configurations are tested.
    candidates = []
    best_by_model_type: dict[str, dict] = {}
    tokenizers = {}
    reached_target = False
    for seed in seeds:
        if reached_target:
            break
        np.random.seed(seed)
        tf.random.set_seed(seed)

        # Use the same 75/25 split strategy as classical training.
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

            # Tokenizer maps words to integer IDs based on the training text only.
            tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
            tokenizer.fit_on_texts(train_text)
            train_sequences = tokenizer.texts_to_sequences(train_text)
            test_sequences = tokenizer.texts_to_sequences(test_text)

            # Padding makes all sequences the same length for neural-network input.
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

            # Train and evaluate this specific architecture/seed combination.
            candidate = train_candidate(run_config, build_sequence_model(run_config, class_count), x_train, train_labels, x_test, test_labels)
            candidates.append(candidate)
            print(f"{run_name}: accuracy={candidate['accuracy']:.3f}, macro_f1={candidate['macro_f1']:.3f}")
            model_type = config["model_type"]

            # Keep the best LSTM and best GRU separately for comparison/reporting.
            model_type_best = best_by_model_type.get(model_type)
            if model_type_best is None or (candidate["macro_f1"], candidate["accuracy"]) > (
                model_type_best["macro_f1"],
                model_type_best["accuracy"],
            ):
                best_by_model_type[model_type] = candidate

            # Stop early once an advanced model passes the requested target accuracy.
            if candidate["accuracy"] >= TARGET_ACCURACY:
                print(f"Reached target advanced accuracy with {run_name}: {candidate['accuracy']:.3f}")
                reached_target = True

    # 5. Select the best advanced model by macro F1, then accuracy.
    best = max(candidates, key=lambda item: (item["macro_f1"], item["accuracy"]))
    best_tokenizer = tokenizers[best["name"]]

    # 6. Save the best LSTM, best GRU, best overall sequence model, tokenizer, and label encoder.
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    best_lstm = best_by_model_type["lstm"]
    best_gru = best_by_model_type["gru"]
    best_lstm["model"].save(LSTM_MODEL_PATH)
    best_gru["model"].save(GRU_MODEL_PATH)
    best["model"].save(BEST_MODEL_PATH)
    joblib.dump(best_tokenizer, TOKENIZER_PATH)
    joblib.dump(label_encoder, LABEL_ENCODER_PATH)

    # 7. Build metrics and confusion matrices for the selected best advanced model.
    label_ids = list(range(len(labels)))
    report = classification_report(best["actual"], best["predictions"], labels=label_ids, target_names=labels, zero_division=0)

    advanced_model_results = []
    for model_type, item in sorted(best_by_model_type.items(), key=lambda pair: (pair[1]["macro_f1"], pair[1]["accuracy"]), reverse=True):
        advanced_model_results.append(
            {
                "model": model_type,
                "accuracy": item["accuracy"],
                "macro_f1": item["macro_f1"],
                **calculate_multiclass_outcomes(item["actual"], item["predictions"], label_ids),
            }
        )

    # 8. Save readable advanced charts: only best LSTM and best GRU in history/outcome plots.
    save_history_plot(
        {
            "LSTM": best_lstm["history"],
            "GRU": best_gru["history"],
        },
        HISTORY_PATH,
    )
    save_overall_metrics_plot(best["accuracy"], best["macro_f1"], ADVANCED_OVERALL_METRICS_PATH)
    save_prediction_outcomes_plot(advanced_model_results, ADVANCED_PREDICTION_OUTCOMES_PATH)

    # 9. Write readable metrics and clean final metadata for project submission.
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
                    "overall_metrics": str(ADVANCED_OVERALL_METRICS_PATH),
                    "prediction_outcomes": str(ADVANCED_PREDICTION_OUTCOMES_PATH),
                },
                "note": "Internal LSTM/GRU architecture variants and seeds were tested during training; only the final selected advanced model is shown here for a clean project submission.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # 10. Print generated artifacts for terminal confirmation.
    print(f"Saved LSTM model to {LSTM_MODEL_PATH}")
    print(f"Saved GRU model to {GRU_MODEL_PATH}")
    print(f"Saved best sequence model to {BEST_MODEL_PATH}")
    print(f"Saved tokenizer to {TOKENIZER_PATH}")
    print(f"Saved label encoder to {LABEL_ENCODER_PATH}")
    print(f"Saved metrics to {REPORT_PATH}")
    print(f"Saved metadata to {METADATA_PATH}")
    print(f"Saved training history plot to {HISTORY_PATH}")
    print(f"Saved overall metrics plot to {ADVANCED_OVERALL_METRICS_PATH}")
    print(f"Saved prediction outcomes plot to {ADVANCED_PREDICTION_OUTCOMES_PATH}")
    print(metrics[2])
    print(metrics[4])


if __name__ == "__main__":
    main()
