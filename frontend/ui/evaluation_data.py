"""The evaluation artifacts, read straight off disk.

The API hands every response the deciding engine's headline metrics, which is
what a result needs. The evaluation screen needs the rest: the per-label
cross-source transfer breakdown, the cross-validation folds, the per-class
recall intervals. Those live in the files the training scripts wrote, and the
API does not expose them, so the screen reads them where they are.

Reading, never writing, and never on the request path. If a file is missing the
screen degrades to whatever `/engines` can tell it rather than inventing a
number.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CLASSICAL_METADATA = REPO_ROOT / "models" / "classical" / "model_metadata.json"
ENGINE_METRICS = REPO_ROOT / "models" / "evaluation" / "engine_metrics.json"
DATASET_SOURCES = REPO_ROOT / "backend" / "app" / "data" / "dataset_sources.json"

# The single number this project most wants a reader to see. Published, not
# buried: an evaluation that only reports its best figure is not an evaluation.
TRANSFER_HEADLINE = (
    "Hold an entire CDC or WHO source passage out of training and test on it, and pooled "
    "recall is {value}. The model is substantially learning how one source page words "
    "things, not the underlying condition. Read this before the accuracy."
)


def _read(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


@lru_cache(maxsize=1)
def classical_metadata() -> dict:
    return _read(CLASSICAL_METADATA)


@lru_cache(maxsize=1)
def engine_metrics_file() -> dict:
    return _read(ENGINE_METRICS)


@lru_cache(maxsize=1)
def source_registry() -> dict:
    return _read(DATASET_SOURCES)


@lru_cache(maxsize=1)
def evaluation() -> dict:
    return classical_metadata().get("evaluation", {})


@lru_cache(maxsize=1)
def transfer() -> dict:
    """Cross-source transfer: pooled recall plus the per-label breakdown."""
    return evaluation().get("cross_source_transfer", {})


def pooled_transfer_recall() -> float | None:
    value = transfer().get("pooled_recall")
    return float(value) if isinstance(value, (int, float)) else None


def transfer_for_label(label: str) -> dict | None:
    """What cross-source transfer says about one specific label.

    Used on a result: a prediction of cholera deserves to carry cholera's 0.015
    next to it, not only the pooled figure.
    """
    if not label:
        return None
    entry = transfer().get("per_label", {}).get(label.lower())
    return entry if isinstance(entry, dict) else None


@lru_cache(maxsize=1)
def transfer_rows() -> list[dict]:
    """Per-label transfer, evaluable labels worst-first, then the rest."""
    per_label = transfer().get("per_label", {})
    evaluable, blocked = [], []
    for label, entry in per_label.items():
        if entry.get("evaluable"):
            evaluable.append(
                {
                    "label": label,
                    "recall": float(entry.get("mean_recall_on_held_out_passage", 0.0)),
                    "passages": entry.get("passages"),
                    "evaluable": True,
                }
            )
        else:
            blocked.append(
                {
                    "label": label,
                    "recall": None,
                    "reason": entry.get("reason", "not evaluable"),
                    "evaluable": False,
                }
            )
    evaluable.sort(key=lambda row: row["recall"])
    blocked.sort(key=lambda row: row["label"])
    return evaluable + blocked


@lru_cache(maxsize=1)
def per_class_rows() -> list[dict]:
    rows = evaluation().get("test", {}).get("per_class", [])
    return sorted(rows, key=lambda row: row.get("recall", 0.0))


@lru_cache(maxsize=1)
def source_count() -> int | None:
    """Distinct CDC/WHO captures behind the dataset.

    The registry is keyed by label, and a capture can back more than one label,
    so the count is of distinct URLs rather than of registry entries.
    """
    registry = source_registry()
    if not isinstance(registry, dict):
        return None
    urls = {
        entry.get("url")
        for entries in registry.values()
        if isinstance(entries, list)
        for entry in entries
        if isinstance(entry, dict) and entry.get("url")
    }
    return len(urls) or None


@lru_cache(maxsize=1)
def dataset_facts() -> dict:
    metadata = classical_metadata()
    return {
        "rows": metadata.get("dataset_rows"),
        "labels": metadata.get("label_count"),
        "provenance": metadata.get("dataset_provenance", {}),
        "source_count": source_count(),
    }


@lru_cache(maxsize=1)
def split_facts() -> dict:
    methodology = evaluation().get("methodology", {})
    return {
        "sizes": methodology.get("split_sizes", {}),
        "groups": methodology.get("split_groups", {}),
        "split": methodology.get("split"),
        "model_selection": methodology.get("model_selection"),
        "shipped_model_fit_on": methodology.get("shipped_model_fit_on"),
        "cross_validation": methodology.get("cross_validation"),
        "intervals": methodology.get("intervals"),
        "known_limitations": methodology.get("known_limitations", []),
    }


@lru_cache(maxsize=1)
def cross_validation() -> dict:
    return evaluation().get("cross_validation", {})


@lru_cache(maxsize=1)
def external_use_cases() -> dict:
    return evaluation().get("external_use_cases", {})
