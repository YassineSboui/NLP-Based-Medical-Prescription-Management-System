"""Tests that protect the evaluation from quietly becoming dishonest again.

The previous pipeline produced 0.906 by ranking 72 pipelines on the same 32-row
set it then reported, and shipping a model refit on that set. Nothing in the
repository could have detected that. These tests can.
"""

from __future__ import annotations

import json

from app.core.evaluation import three_way_group_split, load_dataset, wilson_interval
from app.core.paths import CLASSICAL_METADATA_PATH, ENGINE_METRICS_PATH


def test_the_split_never_puts_a_group_on_both_sides():
    """A paraphrase and its parent must not straddle a train/test boundary."""
    split = three_way_group_split(load_dataset())

    train_groups = set(split.train["group_id"])
    validation_groups = set(split.validation["group_id"])
    test_groups = set(split.test["group_id"])

    assert not train_groups & test_groups
    assert not train_groups & validation_groups
    assert not validation_groups & test_groups


def test_every_label_appears_in_every_split():
    split = three_way_group_split(load_dataset())
    labels = set(load_dataset()["disease"])

    assert set(split.train["disease"]) == labels
    assert set(split.validation["disease"]) == labels
    assert set(split.test["disease"]) == labels


def test_the_split_is_deterministic():
    first = three_way_group_split(load_dataset())
    second = three_way_group_split(load_dataset())
    assert list(first.test["row_id"]) == list(second.test["row_id"])


def test_published_metrics_state_their_methodology():
    metadata = json.loads(CLASSICAL_METADATA_PATH.read_text(encoding="utf-8"))
    methodology = metadata["evaluation"]["methodology"]

    assert "group-aware" in methodology["split"]
    assert "validation" in methodology["model_selection"]
    assert "test set is not read" in methodology["model_selection"].lower()
    assert "train + validation" in methodology["shipped_model_fit_on"]
    assert methodology["known_limitations"], "Limitations must be published with the numbers"


def test_headline_accuracy_is_published_with_an_interval():
    metadata = json.loads(CLASSICAL_METADATA_PATH.read_text(encoding="utf-8"))

    assert metadata["accuracy_ci_95"], "A point estimate alone is not honest at this sample size"
    low, high = metadata["accuracy_ci_95"]
    assert low <= metadata["accuracy"] <= high
    assert high - low > 0.01, "An interval this tight on ~120 rows would be suspicious"


def test_per_class_metrics_carry_intervals():
    metadata = json.loads(CLASSICAL_METADATA_PATH.read_text(encoding="utf-8"))
    per_class = metadata["evaluation"]["test"]["per_class"]

    assert len(per_class) == metadata["label_count"]
    for entry in per_class:
        low, high = entry["recall_ci_95"]
        assert low <= entry["recall"] <= high, f"{entry['label']} recall is outside its own interval"


def test_the_generalisation_caveat_is_published_not_buried():
    """Cross-source transfer is much worse than the headline. It must be stated."""
    metadata = json.loads(CLASSICAL_METADATA_PATH.read_text(encoding="utf-8"))
    transfer = metadata["evaluation"]["cross_source_transfer"]

    assert transfer["pooled_recall"] is not None
    assert transfer["per_label"]
    # Labels backed by a single passage cannot be tested this way and must be
    # reported as such rather than folded into the average.
    assert any(entry.get("evaluable") is False for entry in transfer["per_label"].values())


def test_the_confidence_floor_was_chosen_on_validation():
    metadata = json.loads(CLASSICAL_METADATA_PATH.read_text(encoding="utf-8"))
    sweep = metadata["evaluation"]["confidence_floor_sweep"]

    assert "validation" in sweep["measured_on"]
    assert "training split only" in sweep["measured_on"]
    assert len(sweep["rows"]) >= 5

    from app.services.disease_prediction_service import MODEL_CONFIDENCE_FLOOR

    assert any(row["floor"] == MODEL_CONFIDENCE_FLOOR for row in sweep["rows"]), (
        "The floor in use was never measured by the sweep"
    )


def test_engine_metrics_were_all_measured_on_the_same_split():
    document = json.loads(ENGINE_METRICS_PATH.read_text(encoding="utf-8"))
    bases = {
        entry["basis"]
        for entry in document["engines"].values()
        if entry.get("available")
    }
    assert len(bases) == 1, f"Engines were measured on different things: {bases}"

    split = three_way_group_split(load_dataset())
    assert document["split"]["test_rows"] == len(split.test)


def test_wilson_interval_behaves_at_the_boundaries():
    assert wilson_interval(0, 0) == (0.0, 0.0)

    low, high = wilson_interval(10, 10)
    assert high == 1.0
    assert low < 1.0, "A perfect score on 10 items is not certainty"

    low, high = wilson_interval(0, 10)
    assert low == 0.0
    assert high > 0.0

    narrow = wilson_interval(90, 100)
    wide = wilson_interval(9, 10)
    assert (narrow[1] - narrow[0]) < (wide[1] - wide[0]), "More data must give a tighter interval"


def test_metadata_does_not_leak_absolute_author_paths():
    """The old deep_learning_metadata.json shipped C:\\Users\\<name>\\... paths."""
    from app.core.paths import ADVANCED_METADATA_PATH

    for path in (CLASSICAL_METADATA_PATH, ADVANCED_METADATA_PATH, ENGINE_METRICS_PATH):
        if not path.exists():
            continue
        content = path.read_text(encoding="utf-8")
        assert "C:\\\\Users" not in content and "C:/Users" not in content, f"{path.name} leaks a local path"
        assert "/home/" not in content, f"{path.name} leaks a local path"
