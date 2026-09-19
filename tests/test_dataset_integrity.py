"""The dataset's honesty guarantees, re-checked from the CSV alone.

These do not trust the builder. They read the shipped file and verify that
every claim it makes about its own provenance holds, which is the only version
of the check worth having: a bug in build_dataset.py would pass a test that
called build_dataset.py.
"""

from __future__ import annotations

import json

import pytest

from app.core.lexicon import source_forms
from app.core.paths import DATASET_PATH, SCRAPED_SOURCES_PATH

VALID_PROVENANCE = {"source", "source_combination", "paraphrase"}
ATTESTED_PROVENANCE = {"source", "source_combination"}


def split_surface_terms(text: str) -> list[str]:
    """Recover the multi-word surface terms a row's text was built from."""
    known = sorted({form for forms in source_forms().values() for form in forms}, key=len, reverse=True)
    terms: list[str] = []
    remaining = text
    while remaining:
        for surface in known:
            if remaining == surface or remaining.startswith(surface + " "):
                terms.append(surface)
                remaining = remaining[len(surface):].strip()
                break
        else:
            head, _, remaining = remaining.partition(" ")
            terms.append(head)
    return terms


def test_every_row_declares_a_known_provenance(dataset):
    unknown = set(dataset["provenance"]) - VALID_PROVENANCE
    assert not unknown, f"Unrecognised provenance values: {unknown}"


def test_row_ids_are_unique(dataset):
    duplicates = dataset["row_id"][dataset["row_id"].duplicated()].tolist()
    assert not duplicates, f"Duplicate row_id values: {duplicates[:5]}"


def test_attested_rows_are_backed_by_their_own_quote(dataset):
    """The core integrity rule.

    For a row that claims to come from a source, every symptom term in its text
    must appear verbatim in that row's recorded source_quote. A row that fails
    this is a clinical claim attributed to CDC or WHO that they did not make.
    """
    failures: list[str] = []

    for row in dataset[dataset["provenance"].isin(ATTESTED_PROVENANCE)].itertuples():
        quote = str(row.source_quote).lower()
        for term in split_surface_terms(str(row.text)):
            if term not in quote:
                failures.append(f"{row.row_id}: {term!r} is not in its source_quote")

    assert not failures, "Rows claiming source backing that the quote does not support:\n" + "\n".join(
        failures[:15]
    )


def test_paraphrases_point_at_a_real_parent_and_do_not_drift(dataset):
    by_id = {row.row_id: row for row in dataset.itertuples()}
    failures: list[str] = []

    for row in dataset[dataset["provenance"] == "paraphrase"].itertuples():
        parent = by_id.get(row.derived_from)
        if parent is None:
            failures.append(f"{row.row_id}: derived_from {row.derived_from!r} does not exist")
            continue
        if parent.disease != row.disease:
            failures.append(f"{row.row_id}: label differs from parent {row.derived_from}")
        if parent.symptom_terms != row.symptom_terms:
            failures.append(
                f"{row.row_id}: symptom set drifted from parent "
                f"({row.symptom_terms!r} != {parent.symptom_terms!r})"
            )

    assert not failures, "\n".join(failures[:15])


def test_source_rows_never_claim_a_parent(dataset):
    attested = dataset[dataset["provenance"].isin(ATTESTED_PROVENANCE)]
    assert attested["derived_from"].fillna("").eq("").all(), (
        "A source-backed row must not be derived from another row"
    )


def test_a_paraphrase_shares_its_parents_group(dataset):
    """Otherwise the evaluation split can put near-duplicates on both sides."""
    groups = dict(zip(dataset["row_id"], dataset["group_id"]))
    for row in dataset[dataset["provenance"] == "paraphrase"].itertuples():
        assert row.group_id == groups[row.derived_from], (
            f"{row.row_id} is in a different group from its parent, which would leak "
            f"a near-duplicate across a train/test boundary"
        )


def test_every_cited_url_is_https_and_from_a_public_health_source(dataset):
    allowed_hosts = ("https://www.cdc.gov/", "https://www.who.int/")
    bad = sorted({url for url in dataset["source_url"] if not str(url).startswith(allowed_hosts)})
    assert not bad, f"Rows cite sources outside CDC/WHO: {bad}"


def test_cited_urls_were_actually_fetched(dataset):
    """A URL in the dataset must correspond to a real capture in the scrape log."""
    records = json.loads(SCRAPED_SOURCES_PATH.read_text(encoding="utf-8"))
    fetched = {
        record["source_url"]
        for record in records
        if record.get("extracted_sections") and record.get("fetch_status") in {"ok", "preserved_previous_capture"}
    }
    missing = sorted(set(dataset["source_url"]) - fetched)
    assert not missing, f"Dataset cites URLs with no successful capture: {missing}"


def test_labels_are_balanced_enough_to_train_on(dataset):
    counts = dataset["disease"].value_counts()
    assert counts.min() >= 20, f"Label with too few rows: {counts.idxmin()} has {counts.min()}"
    assert counts.max() / counts.min() < 2.0, f"Label distribution is too skewed:\n{counts}"


def test_the_dataset_is_substantially_larger_than_the_original(dataset):
    """Guards against a regression to the 128-row lookup table."""
    assert len(dataset) > 400, f"Dataset has shrunk to {len(dataset)} rows"


def test_no_row_is_an_exact_duplicate_of_another(dataset):
    duplicates = dataset[dataset.duplicated(subset=["text", "disease"], keep=False)]
    assert duplicates.empty, f"Duplicate (text, disease) pairs:\n{duplicates[['row_id', 'text']].head()}"


@pytest.mark.parametrize("column", ["row_id", "text", "disease", "provenance", "group_id", "source_url"])
def test_required_columns_are_never_empty(dataset, column):
    empty = dataset[dataset[column].isna() | dataset[column].astype(str).str.strip().eq("")]
    assert empty.empty, f"Column {column!r} is empty on rows: {empty['row_id'].tolist()[:5]}"


def test_dataset_csv_path_exists():
    assert DATASET_PATH.exists()
